from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(resume_emb, jd_emb):
    # Convert CUDA tensors to CPU NumPy arrays
    resume_np = resume_emb.cpu().numpy()
    jd_np = jd_emb.cpu().numpy()

    score = cosine_similarity([resume_np], [jd_np])[0][0]
    return round(score, 3)