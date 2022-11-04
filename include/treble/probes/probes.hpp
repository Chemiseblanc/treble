#ifndef TREBLE_PROBES_H
#define TREBLE_PROBES_H

/*
/// @brief A probe class is used to evaluate the objective function for an optimizer.
/// It begins measuring some quantity on creation, and on destruction stops measuring
/// and passes the measurement as the current value of the objective function to the optimizer
/// through the provided callback.
class Probe {
 public:
  using measure_type = int;

  explicit Probe(std::function<void(measure_type)> callback);
  Probe(const Probe &);
  Probe(Probe &&);
  ~Probe();

  Probe &operator=(const Probe &) = default;
  Probe &operator=(Probe &&) = default;

 private:
  std::function<void(measure_type)> on_exit;
};
*/

#include "treble/probes/scoped_timer.hpp"

#endif  // TREBLE_PROBES_H