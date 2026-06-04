
#ifndef STAT_METHODS_H
#define STAT_METHODS_H

#include <cassert>
#include <cstdint>

#include "msl/bits.h"
#include "msl/fwcounter.h"

template <typename T>
struct category_projector {
  auto operator()(const T& t) const { return t; }
};

namespace champsim::msl
{

/**
 * Splits an input space into categories for set dueling.
 *
 * Given a candidate value and a sample rate, the categorizer assigns the
 * candidate to a category number. Category 0 and category 1 are the two
 * "leader" categories used by set-dueling policies like DRRIP and SHiP;
 * all other categories are "follower" sets that use the winning policy.
 *
 * The sample rate must be a power of 2 and determines how many categories
 * exist (equal to the sample rate).
 *
 * \tparam T The type of value being categorized (e.g. ``long`` for set indices).
 * \tparam CatProj A function object that projects a value of type T to an integer. Defaults to identity.
 */
template <typename T, typename CatProj = category_projector<T>>
class categorizer
{
private:
  std::size_t sample_rate;
  CatProj cat_projection;

public:
  /** Return the sample rate used by this categorizer. */
  std::size_t get_sample_rate() const { return sample_rate; }
  /**
   * Return the category number for a given candidate.
   *
   * Category 0 and 1 are the two leader categories. All other values
   * are follower categories.
   *
   * \param candidate The value to categorize.
   * \return The category index (0 to sample_rate - 1).
   */
  std::size_t get_sample_category(const T& candidate) const
  {
    auto sp = cat_projection(candidate);
    champsim::data::bits shift{lg2(sample_rate)};
    auto mask = bitmask(shift);

    auto low_slice = sp & mask;
    auto high_slice = (sp >> lg2(sample_rate)) & mask;
    return (sample_rate + low_slice - high_slice) & mask;
  }
  categorizer(std::size_t sample_rate_, CatProj cat_projection_) : sample_rate(sample_rate_), cat_projection(cat_projection_) {}
  /** Construct a categorizer with the given sample rate and default projection. */
  categorizer(std::size_t sample_rate_) : categorizer(sample_rate_, CatProj{}) {}
};

/**
 * A dueling saturating counter for set-dueling policies.
 *
 * Combines a categorizer with a fixed-width saturating counter to implement
 * set dueling. Category 0 samples are treated as "policy A" and category 1
 * samples as "policy B". The counter tracks which policy is performing better.
 *
 * Use ``decide()`` to choose a policy for follower sets, ``update_good()`` when
 * a sample performs well, and ``update_bad()`` when a sample performs poorly.
 *
 * Used by replacement policies such as DRRIP and SHiP.
 *
 * \tparam T The type of value being categorized (e.g. ``long`` for set indices).
 * \tparam COUNTER_WIDTH The bit width of the internal saturating counter.
 * \tparam CatProj A function object that projects a value of type T to an integer. Defaults to identity.
 */
template <typename T, std::size_t COUNTER_WIDTH, typename CatProj = category_projector<T>>
class dscounter
{
private:
  categorizer<T, CatProj> cat_sampler;
  fwcounter<COUNTER_WIDTH> counter;

public:
  /** Return the sample rate used by the internal categorizer. */
  std::size_t get_sample_rate() const { return cat_sampler.get_sample_rate(); }

  /**
   * Choose a policy for the given candidate.
   *
   * Returns ``true`` for category-0 candidates (always use policy A),
   * ``false`` for category-1 candidates (always use policy B), and for
   * follower categories returns ``true`` if policy A is currently winning.
   *
   * \param candidate The value to decide on.
   * \return ``true`` if policy A should be used.
   */
  bool decide(const T& candidate)
  {
    auto category = cat_sampler.get_sample_category(candidate);
    if (category == 0)
      return true;
    else if (category == 1)
      return false;
    return counter >= (counter.maximum / 2);
  }

  /**
   * Report a good outcome for the given candidate.
   *
   * Increments the counter for category-0 samples and decrements it for
   * category-1 samples. Has no effect on follower categories.
   *
   * \param candidate The value that had a good outcome.
   */
  void update_good(const T& candidate)
  {
    auto category = cat_sampler.get_sample_category(candidate);
    if (category == 0) {
      counter += 1;
    } else if (category == 1) {
      counter -= 1;
    }
  }
  /**
   * Report a bad outcome for the given candidate.
   *
   * Decrements the counter for category-0 samples and increments it for
   * category-1 samples. Has no effect on follower categories.
   *
   * \param candidate The value that had a bad outcome.
   */
  void update_bad(const T& candidate)
  {
    auto category = cat_sampler.get_sample_category(candidate);
    if (category == 0) {
      counter -= 1;
    } else if (category == 1) {
      counter += 1;
    }
  }
  dscounter(std::size_t sample_rate_, CatProj cat_projection_) : cat_sampler(sample_rate_, cat_projection_), counter(0) {}
  dscounter(std::size_t sample_rate_) : dscounter(sample_rate_, CatProj{}) {}
};

/**
 * Determine the sampling rate for set dueling based on the number of sets.
 *
 * Returns a power-of-2 sample rate: 32 for 1024+ sets, 16 for 256--1023,
 * 8 for 64--255, 4 for 8--63. Asserts if fewer than 8 sets.
 *
 * \param num The total number of sets.
 * \return The sample rate (a power of 2).
 */
static inline std::size_t get_sample_rate(long num)
{
  std::size_t set_sample_rate = 32; // 1 in 32
  if (num < 1024 && num >= 256) {   // 1 in 16
    set_sample_rate = 16;
  } else if (num >= 64) { // 1 in 8
    set_sample_rate = 8;
  } else if (num >= 8) { // 1 in 4
    set_sample_rate = 4;
  } else {
    assert(false); // Not enough sets to sample for set dueling
  }
  return set_sample_rate;
}
/**
 * Calculate the number of samples for set dueling.
 *
 * Returns ``num / get_sample_rate(num)``. Asserts that the number of sets
 * is evenly divisible by the sample rate.
 *
 * \param num The total number of sets.
 * \return The number of samples per category.
 */
static inline std::size_t get_num_samples(long num)
{
  assert(num % get_sample_rate(num) == 0);
  return num / get_sample_rate(num);
}
} // namespace champsim::msl

#endif