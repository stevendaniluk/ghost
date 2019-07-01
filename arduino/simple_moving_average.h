#pragma once

/* SimpleMovingAverage
 *
 * Computes and stores a simple moving average over a fixed window size.
 *
 * The template parameter N is the window size.
 */
template <typename DataT, uint8_t N>
class SimpleMovingAverage {
  public:
    /* SimpleMovingAverage
     *
     * @param init_value: Value to initialize the moving average to
     */
    SimpleMovingAverage(const DataT& init_value = 0) { reset(init_value); }

    /* reset
     *
     * @param new_value: Value to assign to the current average and all values in the window
     */
    void reset(const DataT& new_value = 0) {
        average_ = new_value;

        index_ = 0;
        for (uint8_t i = 0; i < N; ++i) {
            data_[i] = new_value;
        }
    }

    /* addDataPoint
     *
     * @param new_point: New data value to add to the moving average
     */
    void addDataPoint(const DataT& new_point) {
        // Get the oldest data point in the history, replace it with the newest, and update the
        // index for out position in the window
        const DataT oldest = data_[index_];
        data_[index_] = new_point;
        index_++;
        if (index_ == N) {
            index_ = 0;
        }

        // Can perform an incremental update based on the new and oldest points in the window
        average_ += (static_cast<float>(new_point) - static_cast<float>(oldest)) / N;
    }

    /* getAverage
     *
     * @return: Current value of the moving average
     */
    DataT getAverage() const { return static_cast<DataT>(average_); }

  private:
    // All data in the window
    DataT data_[N];
    uint8_t index_;
    // Current average value
    float average_;
};  // end SimpleMovingAverage
