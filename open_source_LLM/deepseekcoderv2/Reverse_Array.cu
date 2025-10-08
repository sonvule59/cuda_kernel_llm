#include <iostream>
#include <vector>

void reverseArray(std::vector<float>& arr) {
    int start = 0;
    int end = arr.size() - 1;

    while (start < end) {
        std::swap(arr[start], arr[end]);
        ++start;
        --end;
    }
}

int main() {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    reverseArray(input);

    for (const auto& num : input) {
        std::cout << num << " ";
    }

    return 0;
}