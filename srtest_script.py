from movie_analysis import MovieAnalyzer

analyzer = MovieAnalyzer()

# Get top 10 movie genres
print(analyzer.movie_type(10))

# Get actor count histogram
print(analyzer.actor_count())

# Get filtered actor distribution and plot
analyzer.actor_distributions(gender="Male", min_height=150, max_height=190, plot=True)
