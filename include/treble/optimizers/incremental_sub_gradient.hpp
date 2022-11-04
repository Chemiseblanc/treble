#ifndef TREBLE_OPTIMIZER_INC_SUB_GRAD_H
#define TREBLE_OPTIMIZER_INC_SUB_GRAD_H

#include <chrono>

#include "treble/detail.hpp"
#include "treble/param.hpp"
#include "treble/probes/scoped_timer.hpp"

namespace treble {

template <typename Probe, typename... BoundArgs>
class IncrementalSubGradientImpl {
 public:
  using tuple_type = std::tuple<std::decay_t<BoundArgs>...>;
  using state_type =
      typename detail::state_vector<tunable_param, BoundArgs...>::type;
  using probe_type = Probe;
  using measure_type = typename probe_type::measure_type;

 private:
  tuple_type &bound_arguments;
  state_type stored_state{copy_current_state()};
  constexpr static size_t nb_optimization_vars =
      std::tuple_size<state_type>::value;
  size_t iteration = 0;
  measure_type starting_value;
  std::array<measure_type, 2 * nb_optimization_vars> trial_values;

 public:
  constexpr explicit IncrementalSubGradientImpl(tuple_type &args) noexcept
      : bound_arguments{args} {};

  constexpr probe_type make_probe() noexcept {
    return probe_type{
        [this](measure_type sample) { evaluate(std::move(sample)); }};
  }

 private:
  /**
   * @brief Write new values to the tunable parameters in the stored function
   *  argument list
   */
  constexpr void update_state(state_type &state) noexcept {
    detail::scatter_array_to_tuple(bound_arguments, state);
  }

  /**
   * @brief Copy the tunable paramters from the stored function argument list
   *  into an array
   */
  constexpr state_type copy_current_state() noexcept {
    return detail::gather_array_from_tuple<tunable_param>(bound_arguments);
  }
  /**
   * @brief Entrypoint into the optimization routine. This is called once after
   *  each return of the wrapped function.
   *
   * Implements a technique similar to an incremental subgradient method for
   * optimizing non-differentialble functions.
   */
  constexpr void evaluate(measure_type sample) noexcept {
    // Save the timing information
    iteration == 0 ? starting_value = std::move(sample)
                   : trial_values.at(iteration - 1) = std::move(sample);

    if (iteration > 0 && iteration % trial_values.size() == 0) {
      // Run single iteration of discrete gradient descent after
      // adjacent locations of the parameter space have been sampled
      state_type new_state = stored_state;
      for (size_t i = 0; i < nb_optimization_vars; ++i) {
        tunable_param &param = new_state[i];
        size_t bw_idx = 2 * i;
        size_t fw_idx = 2 * i + 1;

        measure_type &bw_value = trial_values.at(bw_idx);
        measure_type &fw_value = trial_values.at(fw_idx);

        if (bw_value < fw_value) {
          if (bw_value < starting_value) {
            param.value = std::max(param.min, param.value - param.step);
          }
        } else {
          if (fw_value < starting_value) {
            param.value = std::min(param.max, param.value + param.step);
          }
        }
      }
      stored_state = std::move(new_state);
      update_state(stored_state);
      iteration = 0;
    } else {
      // Sample the next location in the parameter space
      // The format of trial_values is
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

struct IncrementalSubGradient {
  template <typename... Ts>
  static constexpr auto make(Ts &&...ts) noexcept {
    return IncrementalSubGradientImpl<scoped_timer<std::chrono::nanoseconds>,
                                      Ts...>{
        std::make_tuple(std::forward<Ts>(ts)...)};
  }
};
}  // namespace treble

#endif  // TREBLE_OPTIMIZER_INC_SUB_GRAD_H