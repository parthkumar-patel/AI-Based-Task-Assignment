from faker import Faker
import random
import json

fake = Faker()

technologies = ['React', 'Node.js', 'MongoDB', 'Docker', 'OAuth2', 'JWT', 'Redis', 'Jest']
departments = ['Frontend', 'Backend', 'DevOps', 'QA']

def generate_employee(num):
    return {
        'employee_id': f'E{num}',
        'name': fake.name(),
        'skills': random.sample(technologies, k=random.randint(2,4)),
        'capacity': random.randint(3, 8), # Hours/day
        'current_load': 0,
        'past_projects': [
            {
                'project_id': f'P{random.randint(100,999)}',
                'role': random.choice(['Lead', 'Developer', 'Architect']),
                'tech_used': random.sample(technologies, k=2)
            } for _ in range(random.randint(1,3))
        ]
    }

def generate_task(num):
    return {
        'task_id': f'T{num}',
        'description': f"Implement {random.choice(['auth module', 'payment gateway', 'CI pipeline'])}",
        'required_skills': random.sample(technologies, k=2),
        'estimated_hours': random.choice([8, 16, 24, 40]),
        'dependencies': [],
        'status': 'Pending'
    }

# Generate dataset
dataset = {
    'employees': [generate_employee(i) for i in range(1,6)],
    'tasks': [generate_task(i) for i in range(1,5)],
    'prd': "Build secure authentication system with OAuth2 and JWT support including:\n- User registration flow\n- Password reset functionality\n- Session management\n- Role-based access control",
    'company_docs': [
        'API Security Standards v2.3',
        'Microservices Architecture Guidelines',
        'Sprint 24-Q2 Retrospective Report',
        'Identity Service Design Spec'
    ]
}

print(json.dumps(dataset, indent=2))
