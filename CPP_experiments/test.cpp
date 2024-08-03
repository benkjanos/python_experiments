#include <iostream>
#include <stack>
#include <string>
#include <thread>
#include <mutex>

class CallStack {
public:
    void push(const std::string& function) {
        std::lock_guard<std::mutex> guard(mutex_);
        stack_.push(function);
    }

    void pop() {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!stack_.empty()) {
            stack_.pop();
        }
    }

    void print() {
        std::lock_guard<std::mutex> guard(mutex_);
        std::stack<std::string> copy = stack_;
        while (!copy.empty()) {
            std::cout << copy.top() << std::endl;
            copy.pop();
        }
    }

private:
    std::stack<std::string> stack_;
    std::mutex mutex_;
};

thread_local CallStack callStack;

class CallStackGuard {
public:
    CallStackGuard(const std::string& function) : function_(function) {
        callStack.push(function_);
    }

    ~CallStackGuard() {
        callStack.pop();
    }

private:
    std::string function_;
};

#define TRACE_FUNCTION CallStackGuard guard(__FUNCTION__)

void foo() {
    TRACE_FUNCTION;
    // Function body
}

void bar() {
    TRACE_FUNCTION;
    foo();
}

int main() {
    bar();
    callStack.print();
    return 0;
}