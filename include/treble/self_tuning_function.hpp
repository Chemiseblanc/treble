#ifndef TREBLE_SELF_TUNING_FUNCTION_H
#define TREBLE_SELF_TUNING_FUNCTION_H
//          Copyright Matthew Gibson 2022.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <functional>
#include <type_traits>
#include <utility>

#include "treble/detail.hpp"
#include "treble/scoped_timer.hpp"

namespace treble {
namespace detail {
/**
 * @brief Uses SFINAE overload selection to turn a stored placeholder or tuning
 * parameter into the real value to be passed to a function
 *
 * @tparam T
 * @tparam PlaceholderArgs
 */
template <typename TuningParam, typename... PlaceholderArgs>
struct sub_placeholder_by_value {
  std::tuple<PlaceholderArgs &&...> args;

  explicit sub_placeholder_by_value(PlaceholderArgs &&...func_args) noexcept
      : args{std::forward<PlaceholderArgs>(func_args)...} {}

  template <typename U,
            std::enable_if_t<std::is_same_v<TuningParam, std::decay_t<U>>,
                             bool> = true>
  constexpr decltype(auto) operator[](U &tuning_param) noexcept {
    return tuning_param.value;
  }

  template <typename Placeholder,
            std::enable_if_t<std::is_placeholder_v<Placeholder>, bool> = true>
  constexpr decltype(auto) operator[](Placeholder) noexcept {
    return std::get<std::is_placeholder<Placeholder>::value - 1>(
        std::move(args));
  }
};
}  // namespace detail

/**
 * @brief Aggregate type for holding parameter information
 */
struct tunable_param {
  int value;
  int min;
  int max;
  int step;
};

/**
 * @brief Function object that creates a std::bind like interface for turning a
 * function into a self-tuning function for integer parameters
 */
template <typename Duration, typename Callable, typename... Args>
class self_tuning_function {
 public:
  using duration_type = Duration;

  /**
   * @brief Create the wrapper object and store the callable object and a copy
   *  of the passed in arguments
   */
  constexpr explicit self_tuning_function(Callable &&func,
                                          Args &&...args) noexcept
      : func{std::forward<Callable>(func)},
        arguments{std::forward<Args>(args)...} {}

  /**
   * @brief Calls the wrapped function, passing through arguments for the
   * placeholder values and return values. Measures the time for the function to
   * execute and passes it to the optimization entry-point
   */
  template <typename... FreeArgs>
  constexpr decltype(auto) operator()(FreeArgs &&...args) noexcept(
      noexcept(call(std::make_index_sequence<sizeof...(Args)>{},
                    std::forward<FreeArgs>(args)...))) {
    scoped_timer<Duration> timer{
        [this](Duration duration) { update_parameters(std::move(duration)); }};
    return call(std::make_index_sequence<sizeof...(Args)>{},
                std::forward<FreeArgs>(args)...);
  }

 private:
  std::decay_t<Callable> func;
  std::tuple<std::decay_t<Args>...> arguments;

  constexpr const static size_t nb_optimization_vars =
      detail::type_count<tunable_param, Args...>::value();
  using state_type = std::array<tunable_param, nb_optimization_vars>;
  state_type stored_state{copy_current_state()};

  size_t iteration = 0;
  Duration starting_time;
  std::array<Duration, 2 * nb_optimization_vars> trial_times;

  /**
   * @brief Call the wrapped function, replacing the placeholder arguments with
   *  the corresponding arguments passed to this function and the tuning
   *  parameters with their current value
   */
  template <typename... FreeArgs, size_t... Seq>
  constexpr decltype(auto)
  call(std::index_sequence<Seq...>, FreeArgs &&...args) noexcept(noexcept(
      func((detail::sub_placeholder_by_value<tunable_param, FreeArgs...>{
          std::forward<FreeArgs>(args)...}[std::get<
          std::integral_constant<size_t, Seq>::value>(arguments)])...))) {
    return func((detail::sub_placeholder_by_value<tunable_param, FreeArgs...>{
        std::forward<FreeArgs>(args)...}[std::get<
        std::integral_constant<size_t, Seq>::value>(arguments)])...);
  }

  /**
   * @brief Write new values to the tunable parameters in the stored function
   *  argument list
   */
  constexpr void update_state(state_type &state) noexcept {
    detail::scatter_array_to_tuple(arguments, state);
  }

  /**
   * @brief Copy the tunable paramters from the stored function argument list
   *  into an array
   */
  constexpr state_type copy_current_state() noexcept {
    return detail::gather_array_from_tuple<tunable_param>(arguments);
  }

  /**
   * @brief Entrypoint into the optimization routine. This is called once after
   *  each return of the wrapped function.
   *
   * Implements a technique similar to an incremental subgradient method for
   * optimizing non-differentialble functions.
   */
  void update_parameters(const Duration duration) noexcept {
    // Save the timing information
    iteration == 0 ? starting_time = std::move(duration)
                   : trial_times.at(iteration - 1) = std::move(duration);

    if (iteration > 0 && iteration % trial_times.size() == 0) {
      // Run single iteration of discrete gradient descent after
      // adjacent locations of the parameter space have been sampled
      state_type new_state = stored_state;
      for (size_t i = 0; i < nb_optimization_vars; ++i) {
        tunable_param &param = new_state[i];
        size_t bw_idx = 2 * i;
        size_t fw_idx = 2 * i + 1;

        Duration &bw_time = trial_times.at(bw_idx);
        Duration &fw_time = trial_times.at(fw_idx);

        if (bw_time < fw_time) {
          if (bw_time < starting_time) {
            param.value = std::max(param.min, param.value - param.step);
          }
        } else {
          if (fw_time < starting_time) {
            param.value = std::min(param.max, param.value + param.step);
          }
        }
      }
      stored_state = std::move(new_state);
      update_state(stored_state);
      iteration = 0;
    } else {
      // Sample the next location in the parameter space
      // The format of trial_times is
      // [param 1 backward step, param 1 forward step, param 2 backward step...]
      enum { BACKWARD = 0, FORWARD = 1 };
      size_t param_idx = iteration / 2;
      size_t param_dir = iteration % 2;

      state_type trial_state = stored_state;
      tunable_param &param = trial_state[param_idx];
      if (param_dir == BACKWARD) {
        param.value = std::max(param.min, param.value - param.step);
      } else if (param_dir == FORWARD) {
        param.value = std::min(param.max, param.value + param.step);
      }

      update_state(trial_state);
      ++iteration;
    }
  }
};

/**
 * @brief Factory function to provide a clean interface for creating a new
 * self tuning function wrapper.
 */
template <typename Duration = std::chrono::microseconds, typename Callable,
          typename... Args>
[[nodiscard]] constexpr self_tuning_function<Duration, Callable, Args...>
self_tuning(Callable &&callable, Args &&...args) noexcept {
  return self_tuning_function<Duration, Callable, Args...>{
      std::forward<Callable>(callable), std::forward<Args>(args)...};
}
}  // namespace treble

#endif  // TREBLE_SELF_TUNING_FUNCTION_H