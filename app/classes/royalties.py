import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def generate_dummy_data(num_rows):
    columns = ["Artist 1", "Artist 2", "Cowriter 1", "Cowriter 2", "Producer", "Publisher"]
    df = pd.DataFrame(columns=columns)

    for _ in range(num_rows):
        random_numbers = np.random.random(len(columns))
        normalized_numbers = random_numbers / random_numbers.sum()
        df.loc[len(df)] = normalized_numbers

    return df

dummy_df = generate_dummy_data(10)
dummy_df = dummy_df.round(2)  # round to 2 decimal places

# Turn the index into "Song 1", "Song 2", etc.
dummy_df.index = ["Song {}".format(i) for i in range(1, len(dummy_df) + 1)]

st.write(dummy_df)

# Create a stacked bar chart using the dummy data
fig = px.bar(dummy_df, barmode="stack")

# Add a title to the plotly figure
fig.update_layout(title_text="Royalties by Song")

# The label for the x axis should be "Songs", and the y-axis should be "Royalties"
fig.update_xaxes(title_text="Songs")
fig.update_yaxes(title_text="Royalties")

# Display the plotly figure using st.plotly_chart()
st.plotly_chart(fig)

# Create a pie chart using the total royalties for each artist
# Hint: use the .sum() method on the dataframe
fig = px.pie(dummy_df.sum(), values=0, names=dummy_df.sum().index)

# Add a title to the plotly figure
fig.update_layout(title_text="Royalties by Stakeholder")

# Display the plotly figure using st.plotly_chart()
st.plotly_chart(fig)