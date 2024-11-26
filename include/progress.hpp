#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdio>

class ProgressBar {
public:
    ProgressBar(int total, int barWidth = 50)
        : total_(total), barWidth_(barWidth), current_(0), stop_(false) {
        hideCursor();
        // Start the progress bar thread
        thread_ = std::thread(&ProgressBar::run, this);
    }

    ~ProgressBar() {
        finish(); // Ensure progress bar finishes and cursor is shown
    }

    void update(int progress) {
        current_ = progress;
        if (progress > total_) current_ = total_;
    }

    void finish() {
        stop_ = true;
        if (thread_.joinable()) {
            thread_.join(); // Wait for the thread to finish
        }
        // Ensure the bar shows 100%
        update(total_);
        display(); // Final display call to ensure it shows 100%
        showCursor();
        std::cout << std::flush; // Flush without printing a new line
    }

private:
    void run() {
        while (!stop_) {
            display();
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Sleep to reduce CPU usage
        }
    }

    void display() {
        float fraction = static_cast<float>(current_) / total_;
        int pos = static_cast<int>(barWidth_ * fraction);

        // Output the progress bar (overwrite the existing line)
        std::cout << "\r["; // Carriage return to start of line
        for (int i = 0; i < barWidth_; ++i) {
            if (i < pos) std::cout << "\033[32m=\033[0m"; // Green =
            else if (i == pos) std::cout << "\033[32m>\033[0m"; // Green >
            else std::cout << " ";
        }
        std::cout << "] " << int(fraction * 100) << " %" << std::flush; // Flush output without a new line
    }

    void hideCursor() {
        std::cout << "\033[?25l"; // ANSI escape code to hide the cursor
        std::cout.flush();
    }

    void showCursor() {
        std::cout << "\033[?25h"; // ANSI escape code to show the cursor
        std::cout.flush();
    }

    int total_;
    int barWidth_;
    std::atomic<int> current_;
    std::atomic<bool> stop_;
    std::thread thread_;
};
