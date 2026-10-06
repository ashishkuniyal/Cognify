import pandas as pd
import numpy as np
import os

def generate_demo_dataset():
    input_path = 'notebook/data/stud.csv'
    output_path = 'notebook/data/stud_demo.csv'
    
    print("Loading original dataset...")
    df = pd.read_csv(input_path)
    
    np.random.seed(42)
    
    # 1. Base the synthetic features slightly on existing scores so they correlate realistically
    # We create a hidden "underlying ability" score just for synthesis
    base_ability = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1) / 100.0
    
    print("Generating synthetic behavioral features...")
    # Attendance: heavily correlated with score, generally high
    # 60 to 100 range, centered around ability
    df['attendance_rate'] = np.clip(np.random.normal(loc=60 + (35 * base_ability), scale=8), 40, 100).round(1)
    
    # Study hours per week: 0 to 20
    df['study_hours_per_week'] = np.clip(np.random.normal(loc=2 + (15 * base_ability), scale=3), 0, 25).round(1)
    
    # Previous GPA: 1.0 to 4.0
    df['previous_gpa'] = np.clip(np.random.normal(loc=1.5 + (2.5 * base_ability), scale=0.4), 1.0, 4.0).round(2)
    
    # Assignment completion rate: 0 to 100
    df['assignment_completion_rate'] = np.clip(np.random.normal(loc=40 + (55 * base_ability), scale=10), 0, 100).round(1)
    
    print("Defining academic risk target...")
    # We define academic risk based on overall performance and engagement
    # Since we want to predict math/reading/writing AND risk, risk will be a derived label
    # A student is HIGH RISK if average score < 60 OR attendance < 70 OR previous GPA < 2.0
    # MEDIUM RISK if average score < 75 OR attendance < 85 OR previous GPA < 2.8
    # LOW RISK otherwise
    
    def calculate_risk(row):
        avg_score = (row['math_score'] + row['reading_score'] + row['writing_score']) / 3.0
        
        if avg_score < 60 or row['attendance_rate'] < 70 or row['previous_gpa'] < 2.0:
            return "HIGH"
        elif avg_score < 75 or row['attendance_rate'] < 85 or row['previous_gpa'] < 2.8:
            return "MEDIUM"
        else:
            return "LOW"
            
    df['academic_risk'] = df.apply(calculate_risk, axis=1)
    
    # We also have an overall_score
    df['overall_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1).round(1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Demo dataset created at {output_path}")
    print(df['academic_risk'].value_counts())
    
if __name__ == '__main__':
    generate_demo_dataset()
