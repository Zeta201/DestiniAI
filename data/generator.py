import random
import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()

# Predefined list of locations with fixed latitude and longitude
locations_dict = {
    "New York": (40.712776, -74.005974),
    "Los Angeles": (34.052235, -118.243683),
    "Chicago": (41.878113, -87.629799),
    "Houston": (29.760427, -95.369804),
    "Phoenix": (33.448376, -112.074036),
    "Philadelphia": (39.952583, -75.165222),
    "San Antonio": (29.424122, -98.493629),
    "San Diego": (32.715736, -117.161087),
    "Dallas": (32.776665, -96.796989),
    "San Jose": (37.338208, -121.886329),
    "Austin": (30.267153, -97.743061),
    "Seattle": (47.606209, -122.332069),
    "Denver": (39.739236, -104.990251),
    "Boston": (42.360081, -71.058884),
    "Atlanta": (33.749001, -84.387978),
    "Miami": (25.761680, -80.191790),
    "Orlando": (28.538336, -81.379234),
    "Portland": (45.515232, -122.678448),
    "Minneapolis": (44.977753, -93.265015),
    "Tampa": (27.950575, -82.457177),
    "Indianapolis": (39.768403, -86.158068),
    "Cleveland": (41.499320, -81.694361),
    "Detroit": (42.331427, -83.045753),
    "Las Vegas": (36.169941, -115.139832),
    "Sacramento": (38.581572, -121.494400),
    "Kansas City": (39.099727, -94.578568),
    "Salt Lake City": (40.760779, -111.891047),
    "Columbus": (39.961176, -82.998794),
    "Raleigh": (35.779591, -78.638176)
}


# Generate random users with consistent locations
def generate_users(n=100):
    start_date = pd.to_datetime("2023-01-01")  # Earliest possible signup date
    end_date = pd.to_datetime("2024-04-01")  # Latest possible signup date

    # Sample locations from the predefined list
    location_choices = list(locations_dict.keys())

    data = {
        'user_id': list(range(1, n + 1)),
        'name': [fake.name() for _ in range(n)],
        'age': [random.randint(18, 50) for _ in range(n)],
        'gender': [random.choice(['M', 'F', 'Non-binary']) for _ in range(n)],
        'location': [random.choice(location_choices) for _ in range(n)],
        # Fixed latitude based on location
        'latitude': [locations_dict[loc][0] for loc in random.choices(location_choices, k=n)],
        # Fixed longitude based on location
        'longitude': [locations_dict[loc][1] for loc in random.choices(location_choices, k=n)],
        'interests': [', '.join(random.sample(
            ['music', 'travel', 'sports', 'technology', 'art', 'reading', 'gaming', 'photography', 'cooking', 'fitness'], 3)) for _ in range(n)],
        'about_me': [fake.sentence(nb_words=10) for _ in range(n)],
        'relationship_goal': [random.choice(['Casual Dating', 'Long-term', 'Friends', 'Networking']) for _ in range(n)],
        'personality': [random.choice(['Introvert', 'Extrovert', 'Ambivert']) for _ in range(n)],
        'MBTI': [random.choice(['INFJ', 'ENTP', 'ISTJ', 'ENFP', 'ISFJ', 'ESTP', 'INTP', 'ESFJ']) for _ in range(n)],

        'education_level': [random.choice(['High School', 'Bachelor’s', 'Master’s', 'PhD']) for _ in range(n)],
        'occupation': [fake.job() for _ in range(n)],
        'zodiac_sign': [random.choice(['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra',
                                       'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']) for _ in range(n)],
        'height_cm': [random.randint(150, 200) for _ in range(n)],
        'body_type': [random.choice(['Athletic', 'Average', 'Curvy', 'Slim']) for _ in range(n)],
        'languages_spoken': [', '.join(random.sample(['English', 'Spanish', 'French', 'Mandarin', 'Hindi'], k=random.randint(1, 3))) for _ in range(n)],
        'dating_preference': [random.choice(['Men', 'Women', 'Everyone']) for _ in range(n)],
        'photo_count': [random.randint(1, 6) for _ in range(n)],
        'is_verified': [random.choice([True, False]) for _ in range(n)],
        'is_premium_user': [random.choices([True, False], weights=[0.2, 0.8])[0] for _ in range(n)],
        'daily_active_minutes': [random.randint(0, 120) for _ in range(n)],
        'last_login': [fake.date_between(start_date=pd.to_datetime("2024-03-01"), end_date=pd.to_datetime("2024-04-01")) for _ in range(n)],
        'likes_received': [random.randint(0, 100) for _ in range(n)],
        'likes': [random.sample(range(1, n + 1), random.randint(5, 15)) for _ in range(n)],
        'swipes': [random.randint(10, 500) for _ in range(n)],
        'messages_sent': [random.randint(0, 100) for _ in range(n)],
        'response_rate': [round(random.uniform(0.2, 1.0), 2) for _ in range(n)],
        'signup_date': [fake.date_between(start_date=start_date, end_date=end_date) for _ in range(n)],
    }

    return pd.DataFrame(data)


if __name__ == '__main__':
    # Generate 1000 users as an example
    df_users = generate_users(n=1000)
    df_users.head(1)
    df_users.to_csv('users.csv', index=False)
