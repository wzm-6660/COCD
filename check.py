import json

def clean_knowledge_codes(data, valid_range=(0, 187)):
    for student in data:
        for log in student.get('logs', []):
            # Filter out knowledge codes that are outside the valid range
            log['knowledge_code'] = [k for k in log['knowledge_code'] if valid_range[0] <= k <= valid_range[1]]
    return data

def calculate_statistics(data_file, output_file):
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)

    # Clean the knowledge_code to remove invalid ones
    data = clean_knowledge_codes(data)

    # Save the cleaned data to a new file
    with open(output_file, 'w', encoding='utf8') as out_f:
        json.dump(data, out_f, indent=4, ensure_ascii=False)

    # Initialize sets to avoid counting duplicates
    student_ids = set()
    exercise_ids = set()
    knowledge_concepts = set()

    for student in data:
        student_ids.add(student['user_id'])  # Add student ID to the set
        logs = student.get('logs', [])
        for log in logs:
            exercise_ids.add(log['exer_id'])  # Add exercise ID to the set
            knowledge_concepts.update(log['knowledge_code'])  # Add knowledge concepts to the set

    # Calculate statistics
    num_students = len(student_ids)
    num_exercises = len(exercise_ids)
    num_knowledge_concepts = len(knowledge_concepts)

    return num_students, num_exercises, num_knowledge_concepts


if __name__ == '__main__':
    data_file = '../data/EdNet-1/log_data.json'  # Replace with your actual data file path
    output_file = '../data/EdNet-1/log_data_cleaned.json'  # Path for saving the cleaned data
    num_students, num_exercises, num_knowledge_concepts = calculate_statistics(data_file, output_file)
    print(f'Number of Students: {num_students}')
    print(f'Number of Exercises: {num_exercises}')
    print(f'Number of Knowledge Concepts: {num_knowledge_concepts}')
