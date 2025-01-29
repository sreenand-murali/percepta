import time

class TimedArray:
    def __init__(self, timeout):
        # timeout in seconds (time after which the element will be removed)
        self.timeout = timeout
        self.array = []

    def add(self, element):
        current_time = time.time()
        # Append the element and the current timestamp
        self.array.append((element, current_time))

    def remove_expired_elements(self):
        current_time = time.time()
        # Filter out elements that have expired
        self.array = [item for item in self.array if current_time - item[1] < self.timeout]

    def get_elements(self):
        # Remove expired elements and return the current valid ones
        self.remove_expired_elements()
        return [item[0] for item in self.array]

# Example usage
timed_array = TimedArray(timeout=5)  # Set timeout to 5 seconds

timed_array.add("item1")
time.sleep(2)  # Wait for 2 seconds
timed_array.add("item2")
time.sleep(4)  # Wait for another 4 seconds (total 6 seconds)

print("Current elements:", timed_array.get_elements())  # Should only show "item2" since "item1" is expired
