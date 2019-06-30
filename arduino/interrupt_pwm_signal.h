#pragma once

#include "Arduino.h"

/* InterruptPWMSignal
 *
 * Utility for recording PWM inputs from interrupts.
 */
class InterruptPWMSignal {
  public:
    /* Constructor
     *
     * @param init_pwm: Default pwm value to initialize with
     */
    InterruptPWMSignal(uint16_t init_pwm) {
        pwm_local = init_pwm;
        pwm_shared = init_pwm;
        new_signal = false;
    }

    /* markSignalStart
     *
     * Triggers the beginning of a pwm signal.
     */
    void markSignalStart() {
        // It's a rising edge of the signal pulse, so record its value
        start_time_ = micros();
    }

    /* markSignalEnd
     *
     * Triggers the end of the current signal, and computes the pwm value based on the signal start
     * time.
     */
    void markSignalEnd() {
        // It is a falling edge, so subtract the time of the rising edge to get the pulse duration
        pwm_shared = (uint16_t)(micros() - start_time_);
        // Set the steering flag to indicate that a new steering signal has been received
        new_signal = true;
    }

    /* processNewSignals
     *
     * Updates the local copy of the latest pwm signal. This does not prevent any interrupts from
     * occurring.
     */
    void processNewSignals() {
        if (new_signal) {
            pwm_local = pwm_shared;
            new_signal = false;
        }
    }

    /* processNewSignalsNoInterrupts
     *
     * Disables interrupts while updating the local copy of the latest pwm signal.
     */
    void processNewSignalsNoInterrupts() {
        noInterrupts();
        if (new_signal) {
            pwm_local = pwm_shared;
            new_signal = false;
        }
        interrupts();
    }

    // Local copy of the latest pwm signal, which can be freely accessed
    uint16_t pwm_local;
    // Copy of pwm signal that is set by the interrup
    volatile uint16_t pwm_shared;
    // Flag for when a new signal has been received
    volatile bool new_signal;

  private:
    // Time of the beginning of the interrupt signal
    uint32_t start_time_;
};  // end InterruptSignal
