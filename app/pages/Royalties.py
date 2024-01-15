""" Page to explain the process of the distribution of royalties
determined by the distribution of generated revenue from the cowriter model. """
import streamlit as st
from pathlib import Path
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

st.set_page_config(
    page_title="Royalties",
    page_icon="🎤",
    initial_sidebar_state="auto",
)

if "royalties_page" not in st.session_state:
    st.session_state["royalties_page"] = "home"

def main():
    st.markdown(f"""
    <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
        font-size: 26px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">A new vision for
        equitable distribution in the age of AI.</h4>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">The distribution of royalties in the music industry
        is an incredibly complex system that ensures that everyone involved in the creative process is fairly
        compensated and credited for their contributions to a song.  This could be a single artist, a group of
        artists, a producer, a cowriter, a publisher, and many more.
        </h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">As the industry confronts the new world of generative
        content produced by models trained on an artist's or publisher's catalog, the challenge of maintaining
        the current system of equitable distribution becomes even more complex.
        Companies like Google, OpenAI,
        Apple, and others have begun to approach publishers offering
        to pay for the rights to use their catalogs
        in their training data.  But how in the world would
        this sum be distributed, would it be in the best
        intrest of the parties involved, and how would that even be determined?
        </h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">Additionally, these large tech companies
        have demonstrated that they are not always, if ever, acting in good faith to ensure fairness
        in the training of their models.  They are all famously secretive about where their training data
        comes from, how it is used, and even whether or not they had permission to use it in the first place.
        OpenAI recently admitted it is
        <a href="https://www.theguardian.com/technology/2024/jan/08/ai-tools-chatgpt-copyrighted-material-openai">
        <i>impossible to train their models without copyrighted data</i></a>, an astonishing admission
        given the stakes.  All of the companies that produce these frontier models
        offer some sort of 'copyright protection'
        for the developers using their platforms, essentially saying
        they will cover the legal fees of any developer sued
        for infringement.</h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">Without a clear
        alternative to outright selling or licensing
        their data to these larger companies, publishers may feel trapped,
        essentially surrendering revenue by not consenting, but sacrificing control, protection and the very
        future of royalty distribution with any generative
        content.</h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">First Rule, however, proposes a solution
        that will preserve and protect the industry's best interests without
        sacrificing the ability to capitalize on the 'new pie' of ai generated content.</h3>
    </div>
    <style>
        @keyframes fadeIn {{
            from {{
                opacity: 0;
            }}
            to {{
                opacity: 1;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""<div class="text-container;" style="animation: fadeIn ease 3s;
                -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s;
                -o-animation: fadeIn ease 3s; -ms-animation:
                fadeIn ease 3s;">
                </div>""", unsafe_allow_html=True)

    fr_solution_button = st.button("First Rule's Solution", type = 'primary', use_container_width=True)
    if fr_solution_button:
        st.session_state.royalties_page = "solution"
        st.rerun()

def solution():
    st.markdown(
        f"""<div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
        font-size: 26px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">First Rule's Royalty Solution</h4>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">What if instead of relying on giant tech companies
        to dictate the terms of the use of publishers' data, all stakeholders involved <i>collaborated</i>
        to achieve the best possible outcome for everyone?  What if the publishers could
        <i>retain control</i> of their data, while still being able to capitalize on the new
        opportunities presented by generative content?</h3>
        </h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">At their core, LLMs and other generative models
        get 'smarter' the more data, and the more quality data, they are trained on.  This is why
        the largest tech companies are so eager to acquire as much of it as possible.  But as mentioned
        previously, these models are black boxes... no one outside of these companies knows exactly
        what data is being used and how it is being used, leaving publishers in the dark about
        the true value of their data and unable to negotiate a fair price for it.</h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">This becomes increasingly problematic
        when the incredible complexities of the royalty distribution framework in the music industry becomes
        involved.  But, maybe, it doesn't have to be this way.</h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">If we can assume that the same framework
        that currently governs equity in the industry could be applied to generative models, then
        suddenly it becomes clear that their could be a transparent, and fair path forward.
        If all parties involved know exactly what data was used to train <i>their</i> models, why
        shouldn't the same rules apply?</h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">By calculating the 'percentage of the pie'
        that stakeholders are entitled to from generative content, artists and publishers can
        be certain that credit, compensation, and consent are restored, <i>and</i> that they don't
        miss out on the future of music, which, like it or not, is coming, and fast.</h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">In the next few pages, we demonstrate
        a very basic representation of what such a system may look like in practice.</h3>
    </div>
    <style>
        @keyframes fadeIn {{
            from {{
                opacity: 0;
            }}
            to {{
                opacity: 1;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""<div class="text-container;" style="animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s;
        -o-animation: fadeIn ease 3s; -ms-animation:
        fadeIn ease 3s;">
        </div>""", unsafe_allow_html=True)

    fairness_button = st.button("Fairness in Practice", type = 'primary', use_container_width=True)
    if fairness_button:
        st.session_state.royalties_page = "fairness"
        st.rerun()

def fairness():
    df_songs = pd.read_csv(Path("./resources/song_royalties_data.csv"))
    # Make the "Song Title Column" the index
    df_songs.set_index("Song Title", inplace=True)
    st.markdown(
        f"""<div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
        font-size: 26px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">A Clear Path Forward</h4>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">For simplicity's sake, let's just assume
        that the content produced by a fairly trained model would split evenly among all
        of the stakeholders based on the data used and existing distribution frameworks.
        A dataframe of the breakdown may look something like this, with the percentage owed
        to each party represented as a decimal:</h3>
        </h3>
    </div>
    <style>
        @keyframes fadeIn {{
            from {{
                opacity: 0;
            }}
            to {{
                opacity: 1;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""<div class="text-container;" style="animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s;
        -o-animation: fadeIn ease 3s; -ms-animation:
        fadeIn ease 3s;">
        </div>""", unsafe_allow_html=True)

    dummy_df = generate_dummy_data(10)
    dummy_df = dummy_df.round(2)  # round to 2 decimal places

    # Turn the index into "Song 1", "Song 2", etc.
    dummy_df.index = ["Song {}".format(i) for i in range(1, len(dummy_df) + 1)]

    st.dataframe(dummy_df)

    st.markdown(f"""
    <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
    font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
    -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
    fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">That's a little noisy, though.  Let's
    visualize it with some charts:</h3>
    </div>
    <style>
        @keyframes fadeIn {{
            from {{
                opacity: 0;
            }}
            to {{
                opacity: 1;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)

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

    st.markdown(
        f"""<div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Now of course the actual distribution
        breakdown would be considerably more complex, but by using transparent training
        methods and inviting all stakeholders to the table, any generated content from the resulting
        model could be split automatically with each downstream use case, providing passive revenue
        for all parties, and keeping control where it belongs -- with the artists, publishers, cowriters,
        and the many others who put their heart and soul into each song.</h3>
        </h3>
    </div>
    <style>
        @keyframes fadeIn {{
            from {{
                opacity: 0;
            }}
            to {{
                opacity: 1;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""<div class="text-container;" style="animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s;
        -o-animation: fadeIn ease 3s; -ms-animation:
        fadeIn ease 3s;">
        </div>""", unsafe_allow_html=True)

    royalties_home_button = st.button("Royalties Home", type = 'primary', use_container_width=True)
    if royalties_home_button:
        st.session_state.royalties_page = "home"
        st.rerun()

if st.session_state["royalties_page"] == "home":
    main()
elif st.session_state["royalties_page"] == "solution":
    solution()
elif st.session_state["royalties_page"] == "fairness":
    fairness()
