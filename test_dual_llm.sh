#!/bin/bash

# Script to test dual-llm.py with 20 random prompts
# Uses I/O redirection to send prompts and status commands

# Array of 20 random questions/prompts
prompts=(
    "What is the capital of France?"
    "Explain how photosynthesis works"
    "What are the benefits of renewable energy?"
    "How do you make chocolate chip cookies?"
    "What is the difference between machine learning and AI?"
    "Describe the water cycle"
    "What causes earthquakes?"
    "How does the internet work?"
    "What are the main types of clouds?"
    "Explain the theory of relativity in simple terms"
    "What is the purpose of DNA?"
    "How do vaccines work?"
    "What are the phases of the moon?"
    "Describe how a car engine works"
    "What is climate change?"
    "How do birds fly?"
    "What is the stock market?"
    "Explain how computers process information"
    "What are the different types of government?"
    "How do plants reproduce?"
)

# Create input file with prompts and status commands
input_file="test_input.txt"
rm -f "$input_file"

echo "Creating input file with prompts..."

for i in "${!prompts[@]}"; do
    echo "${prompts[$i]}" >> "$input_file"
    echo "status" >> "$input_file"
done

# Add quit command at the end
echo "quit" >> "$input_file"

echo "Input file created with ${#prompts[@]} prompts"
echo "Running dual-llm.py with input redirection..."
echo "Output will be saved to test_output.log"

# Run the Python script with input redirection using virtual environment
.venv/bin/python dual-llm.py < "$input_file" 2>&1 | tee test_output.log

echo "Test completed. Check test_output.log for results."
