from RatingSystem import RatingSystem
from SVDSystem import SVDSystem
from GenreSystem import GenreSystem

class HybridSystem(RatingSystem):
    def __init__(self, alpha=0.6):
        self.svd = SVDSystem()
        self.genre = GenreSystem()
        self.alpha = alpha

    def rate(self, user, movie):
        pred_svd = self.svd.rate(user, movie)
        pred_genre = self.genre.rate(user, movie)
        
        return self.alpha * pred_svd + (1.0 - self.alpha) * pred_genre

    def __str__(self):
        return f'Hybrid System (alpha={self.alpha})'
