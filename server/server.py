from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from supabase import create_client, Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Initialize Supabase client
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    supabase: Client = create_client(url, key)

    # Retrieve job data from Supabase
    def get_jobs_from_supabase():
        response = supabase.table('jobs_posted').select('job_title, job_description').execute()
        data = response.data
        return pd.DataFrame(data)

    # TF-IDF vectorization
    def update_tfidf(df):
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(df['job_description'])
        return tfidf, tfidf_matrix

    # Cosine Similarity matrix
    def update_cosine_sim(tfidf_matrix):
        return linear_kernel(tfidf_matrix, tfidf_matrix)

    # Define API endpoint for job recommendations
    @app.route("/api/job_recommendations", methods=['POST'])
    def get_job_recommendations():
        # Get fresh data every time the function is called
        df_gigs = get_jobs_from_supabase()
        tfidf, tfidf_matrix = update_tfidf(df_gigs)
        cosine_sim = update_cosine_sim(tfidf_matrix)

        # Extract job title from request data
        data = request.get_json()
        title_to_search = data.get('job_title')

        # Function to get recommendations
        def get_recommendations(title):
            idx = df_gigs.index[df_gigs['job_title'] == title].tolist()[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]  # Exclude the first one, which is the job itself
            recommendations = [df_gigs.iloc[i]['job_title'] for i, _ in sim_scores]
            return recommendations

        # Get recommendations for the given job title
        recommendations = get_recommendations(title_to_search)

        return jsonify({'job_recommendations': recommendations})
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)