#ifndef TREBLE_PARAM_H
#define TREBLE_PARAM_H

namespace treble {
/**
 * @brief Aggregate type for holding parameter information
 */
struct tunable_param {
  int value;
  int min;
  int max;
  int step;

  constexpr tunable_param() : value{}, min{}, max{}, step{} {}
  constexpr tunable_param(int starting_value, int minimum, int maximum,
                          int step_size)
      : value{starting_value}, min{minimum}, max{maximum}, step{step_size} {}
  constexpr tunable_param(int starting_value, int minimum, int maximum)
      : value{starting_value}, min{minimum}, max{maximum}, step{1} {}
  constexpr tunable_param(int minimum, int maximum)
      : value{(minimum + maximum) / 2}, min{minimum}, max{maximum}, step{1} {}
  constexpr tunable_param(const tunable_param&) = default;
  constexpr tunable_param(tunable_param&&) = default;

  constexpr tunable_param& operator=(const tunable_param&) = default;
  constexpr tunable_param& operator=(tunable_param&&) = default;
};
}  // namespace treble

#endif TREBLE_PARAM_H