#pragma once

/* SignalMap
 *
 * Utility for mapping a signal defined by descrete min, max, and center positions to a scalar
 * range.
 */
struct SignalMap {
    /* Constructor
     *
     * @param min_in: Minimum signal position
     * @param center_in: Center signal position
     * @param max_in: Maximum signal position
     */
    SignalMap(uint16_t min_in, uint16_t center_in, uint16_t max_in) {
        min = min_in;
        max = max_in;
        center = center_in;
    }

    /* mapFromUnitInput
     *
     * @param input: Scalar input in the range [-1, 1]
     * @return: input mapped to the signal range for this object (-ve being towards min)
     */
    uint16_t mapFromUnitInput(const float& input) const {
        uint16_t mapped_val;
        if (input >= 0.0) {
            mapped_val = center + input * (max - center);
        } else {
            mapped_val = center + input * (center - min);
        }

        return mapped_val;
    }

    /* mapToUnitOutput
     *
     * @param input: Discrete signal input
     * @return: input mapped to the range [-1, 1] (-ve being towards min)
     */
    float mapToUnitOutput(const uint16_t& input) const {
        float mapped_val;
        if (input >= center) {
            mapped_val = float(input - center) / float(max - center);
        } else {
            mapped_val = -float(center - input) / float(center - min);
        }

        mapped_val = min(max(mapped_val, -1.0), 1.0);
        return mapped_val;
    }

    // Minimum signal value
    uint16_t min;
    // Maximum signal value
    uint16_t max;
    // Center signal value
    uint16_t center;
};  // end SignalMap
