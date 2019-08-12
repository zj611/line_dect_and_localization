# -*- coding: utf-8 -*-
class Stack:
 
    def __init__(self, stack_size):
        self.items = []
        self.stack_size = stack_size
 
    def is_empty(self):
        return len(self.items) == 0
 
    def pop(self):
        return self.items.pop()
 
    def peek(self):
        if not self.is_empty():
            return self.items[len(self.items) - 1]
 
    def size(self):
        return len(self.items)
 
    def push(self, item):
        if self.size() >= self.stack_size:
            for i in range(0,self.size() - self.stack_size + 1):
                self.items.remove(self.items[0])
        self.items.append(item)