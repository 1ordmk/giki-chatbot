"""
Generate realistic GIKI student database for admin features
Run this to create: data/mock_students.json
"""

import json
import random
from pathlib import Path

def generate_student_database(num_students=100):
    """Generate mock student data"""
    
    departments = [
        'Computer Science',
        'Software Engineering', 
        'Electrical Engineering',
        'Mechanical Engineering',
        'Chemical Engineering',
        'Materials Engineering',
        'Management Sciences'
    ]
    
    first_names_male = [
        'Ahmed', 'Ali', 'Hassan', 'Usman', 'Bilal', 'Hamza', 'Omar', 'Faisal',
        'Zain', 'Salman', 'Imran', 'Kamran', 'Shahzad', 'Nadeem', 'Rizwan'
    ]
    
    first_names_female = [
        'Fatima', 'Ayesha', 'Sara', 'Zainab', 'Maryam', 'Hira', 'Aisha',
        'Sana', 'Nida', 'Mahnoor', 'Iqra', 'Rabia', 'Khadija', 'Amna'
    ]
    
    last_names = [
        'Khan', 'Ali', 'Ahmed', 'Shah', 'Malik', 'Hussain', 'Iqbal',
        'Raza', 'Haider', 'Abbasi', 'Butt', 'Mir', 'Syed', 'Qureshi',
        'Chaudhry', 'Sheikh', 'Baig', 'Zaidi', 'Naqvi', 'Rizvi'
    ]
    
    students = {}
    
    for i in range(num_students):
        # Generate student ID (year + 3 digits)
        year = random.choice([2021, 2022, 2023, 2024])
        student_id = f"{year}{random.randint(100, 999)}"
        
        # Avoid duplicates
        while student_id in students:
            student_id = f"{year}{random.randint(100, 999)}"
        
        # Random gender for name selection
        is_male = random.choice([True, False])
        first_name = random.choice(first_names_male if is_male else first_names_female)
        last_name = random.choice(last_names)
        
        # Calculate semester based on year
        current_year = 2025
        years_enrolled = current_year - year
        max_semester = min(years_enrolled * 2, 8)
        semester = random.randint(1, max_semester) if max_semester > 0 else 1
        
        # GPA tends to be higher in later semesters
        base_gpa = random.uniform(2.0, 4.0)
        if semester > 4:
            base_gpa = random.uniform(2.5, 4.0)  # Slight improvement
        
        # Attendance correlation with GPA
        base_attendance = random.uniform(70, 100)
        if base_gpa > 3.5:
            base_attendance = random.uniform(85, 100)
        
        students[student_id] = {
            'name': f"{first_name} {last_name}",
            'department': random.choice(departments),
            'semester': semester,
            'gpa': round(base_gpa, 2),
            'attendance': round(base_attendance, 1),
            'email': f"{student_id}@giki.edu.pk",
            'status': random.choices(
                ['Active', 'On Leave', 'Probation'],
                weights=[90, 5, 5]
            )[0],
            'phone': f"+92-3{random.randint(10, 99)}-{random.randint(1000000, 9999999)}",
            'city': random.choice([
                'Islamabad', 'Rawalpindi', 'Lahore', 'Karachi', 
                'Peshawar', 'Swabi', 'Mardan', 'Abbottabad'
            ])
        }
    
    return students

def main():
    print("\n" + "="*60)
    print("📚 GIKI STUDENT DATABASE GENERATOR")
    print("="*60 + "\n")
    
    # Get number of students
    try:
        num = input("How many students to generate? [default: 100]: ").strip()
        num_students = int(num) if num else 100
    except:
        num_students = 100
    
    print(f"\n🔄 Generating {num_students} student records...")
    students = generate_student_database(num_students)
    
    # Save to file
    output_path = Path('data/mock_students.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(students, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved to: {output_path}")
    
    # Statistics
    print(f"\n📊 Statistics:")
    print(f"   Total students: {len(students)}")
    
    departments = {}
    for sid, student in students.items():
        dept = student['department']
        departments[dept] = departments.get(dept, 0) + 1
    
    print(f"   Departments:")
    for dept, count in sorted(departments.items()):
        print(f"      • {dept}: {count}")
    
    # Show sample students
    print(f"\n📋 Sample Student IDs (for testing):")
    sample_ids = random.sample(list(students.keys()), min(5, len(students)))
    for sid in sample_ids:
        student = students[sid]
        print(f"   • {sid} - {student['name']} ({student['department']})")
    
    print(f"\n✅ Database ready!")
    print(f"\n💡 Test admin queries:")
    print(f"   • 'Get info for student {sample_ids[0]}'")
    print(f"   • 'What is the attendance of {sample_ids[1]}'")
    print(f"   • 'Show details for {sample_ids[2]}'")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()