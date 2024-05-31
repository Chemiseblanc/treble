#ifndef TREBLE_OPTIMIZERS_H
#define TREBLE_OPTIMIZERS_H

/*
/// @brief Optimization algorithm implementation
template<typename Probe, typename... BoundArgs>
class OptimizerImpl {
 public:
  using tuple_type = std::tuple<std::decay_t<BoundArgs>...>;
  using state_type = typename detail::state_vector<tunable_param,
  BoundArgs...>::type; using probe_type = Probe; using measure_type =
  typename Probe::measure_type;

  constexpr explicit OptimizerImpl(tuple_type& args) noexcept :
    bound_arguments{args} {} 
  constexpr probe_type make_probe() noexcept { 
    return probe_type{
        [this](measure_type sample){evaluate(std::move(sample))}};
  }
 private:
  tuple_type& bound_arguments;
  state_type stored_state{copy_current_state()};
  // Extra algorithm-specific state
  constexpr void evaluate(measure_type) noexcept;
  constexpr void update_state(state_type &) noexcept;
  constexpr state_type copy_current_state() noexcept;
};

/// @brief Optimizer algorithm factory struct
/// The purpose of this is to proide a cleaner interface when customizing
/// the optimization routine through the self_tuning, st_front, st_back functions.
/// By using a templated factory function it removes the need to include the full
/// type signature when specifing an alternative as a template parameter
class Optimizer {
    template<typename... Ts>
    static constexpr auto make(Ts&&... ts) noexcept {
      return OptimizerImpl<Ts...>{std::make_tuple(std::forward<Ts>(ts)...)};
    }
};
*/

#include "treble/optimizers/compass_search.hpp"

#endif  // TREBLE_OPTIMIZERS_H
