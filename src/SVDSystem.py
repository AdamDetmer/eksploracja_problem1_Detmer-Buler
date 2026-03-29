import pandas as pd
from surprise import Dataset, Reader, SVD
from RatingSystem import RatingSystem
import test_users
import random

class SVDSystem(RatingSystem):
    def __init__(self, n_factors=20, n_epochs=20, max_train_samples=2500000):
        super().__init__()
        
        print("Inicjalizacja SVDSystem (Surprise)...")
        
        test_set = {tuple(pair) for pair in test_users.test_pairs}
        test_u = {int(p[0]) for p in test_users.test_pairs}
        test_m = {int(p[1]) for p in test_users.test_pairs}
        
        train_data = []
        for u, user_obj in self.users.items():
            for m, r in user_obj.ratings.items():
                if (u, m) not in test_set:
                    # Dodajemy pary (użytkownik, film), które są w zbiorze testowym (ale bez samej oceny testowej)
                    # oraz próbkę pozostałych danych dla szybkości uczenia.
                    if u in test_u or m in test_m:
                        train_data.append({'userID': u, 'itemID': m, 'rating': r})
                    elif random.random() < 0.05:
                        train_data.append({'userID': u, 'itemID': m, 'rating': r})
                        
        # Ograniczenie liczby próbek, aby trening nie trwał zbyt długo - pomyśleć czy to zostawić czy nie jakiś procent?
        if len(train_data) > max_train_samples:
            random.shuffle(train_data)
            train_data = train_data[:max_train_samples]
            
        print(f"SVD Surprise używa {len(train_data)} próbek treningowych")
        
        df = pd.DataFrame(train_data)
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
        
        trainset = data.build_full_trainset()
        
        print(f"Trenowanie modelu Surprise SVD...")
        # Inicjalizacja algorytmu SVD (Matrix Factorization) z biasami
        # n_factors: rozmiar wektorów ukrytych, n_epochs: liczba iteracji, 
        # lr_all: stała uczenia, reg_all: stała regularyzacji
        self.algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=0.005, reg_all=0.05)
        self.algo.fit(trainset)
        print("Trening zakończony.")

    def rate(self, user, movie):
        pred = self.algo.predict(user.id, movie).est
        return pred

    def __str__(self):
        return 'Surprise SVD'
