""" Checking the uploaded song for similarity
to Luke Combs. Create embeddings of Luke Combs
songs to compare uploads or ideas to the artist.
Could filter by lyrics, title, topics, etc.


Example Flow:
    1. User uploads song
    2. Song is converted to embeddings
    3. Song is compared to Luke Combs embeddings
    4. Correspoding metadata is returned to user
    from the vector database. 
    5.  Song is presented to the user, could also
    be fed to the musicgen model for generative 
    ideas."""