#!/usr/bin/env python
# coding: utf-8

# You can download the `requirements.txt` for this course from the workspace of this lab. `File --> Open...`

# # L2: Create Agents to Research and Write an Article
# 
# In this lesson, you will be introduced to the foundational concepts of multi-agent systems and get an overview of the crewAI framework.

# The libraries are already installed in the classroom. If you're running this notebook on your own machine, you can install the following:
# ```Python
# !pip install crewai==0.28.8 crewai_tools==0.1.6 langchain_community==0.0.29
# ```

# In[55]:


# Warning control
import warnings
warnings.filterwarnings('ignore')


# - Import from the crewAI libray.

# In[56]:


from crewai import Agent, Task, Crew


# - As a LLM for your agents, you'll be using OpenAI's `gpt-3.5-turbo`.
# 
# **Optional Note:** crewAI also allow other popular models to be used as a LLM for your Agents. You can see some of the examples at the [bottom of the notebook](#1).

# In[57]:


import os
from utils import get_openai_api_key

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'


# ## Creating Agents
# 
# - Define your Agents, and provide them a `role`, `goal` and `backstory`.
# - It has been seen that LLMs perform better when they are role playing.

# ### Agent: Planner
# 
# **Note**: The benefit of using _multiple strings_ :
# ```Python
# varname = "line 1 of text"
#           "line 2 of text"
# ```
# 
# versus the _triple quote docstring_:
# ```Python
# varname = """line 1 of text
#              line 2 of text
#           """
# ```
# is that it can avoid adding those whitespaces and newline characters, making it better formatted to be passed to the LLM.

# In[58]:


planner = Agent(
    role="Leerplanner",
    goal="Maak een effectief leerplan voor {onderwerp} dat past bij het kennisniveau en leerdoelen van de student",
    backstory="Je bent een ervaren educatieve planner gespecialiseerd in het opstellen van gestructureerde leerpaden. "
              "Je werkt aan het plannen van een leerprogramma over: {onderwerp}. "
              "Je analyseert de complexiteit van het onderwerp en breekt het op in logische, geleidelijke stappen. "
              "Je houdt rekening met verschillende leerstijlen en zorgt voor een gebalanceerde mix van theorie en praktijk. "
              "Je werk vormt de basis voor de Leerbegeleider om de student door het materiaal te loodsen.",
    allow_delegation=False,
    verbose=True
)


# ### Agent: Writer

# In[59]:


uitlegger = Agent(
    role="Didactisch Expert",
    goal="Vertaal complexe concepten over {onderwerp} naar begrijpelijke lessen",
    backstory="Je bent een ervaren docent met talent voor heldere uitleg. "
              "Je gebruikt analogieën, visuals en gestructureerde voorbeelden "
              "om abstracte concepten concreet te maken. Je past je stijl aan "
              "op basis van leerlingvragen en voorkennis.",
    allow_delegation=False,
    verbose=True
)


# ### Agent: Editor

# In[60]:


leercoach = Agent(
    role="Kwaliteitsbewaker",
    goal="Zorg voor effectieve, foutloze en boeiende leercontent",
    backstory="Je bent een scherpe reviewer met didactische ervaring. "
              "Je controleert lessen op wetenschappelijke juistheid, "
              "didactische flow en praktische toepasbaarheid. "
              "Je voegt checkpoints en verbetersleutels toe waar nodig.",
    allow_delegation=False,
    verbose=True
)


# ## Creating Tasks
# 
# - Define your Tasks, and provide them a `description`, `expected_output` and `agent`.

# ### Task: Plan

# In[61]:


leerplan_taak = Task(
    description=(
        "1. Analyseer het onderwerp {onderwerp} en identificeer:\n"
        "   - Kernconcepten en basisprincipes\n"
        "   - Veelgemaakte misvattingen\n"
        "   - Praktische toepassingen\n"
        "   - Relatie met andere onderwerpen\n\n"
        
        "2. Bepaal het geschikte niveau voor de leerling:\n"
        "   - Voorkennis inschatten\n"
        "   - Leerdoelen vaststellen\n"
        "   - Tijdsinvestering bepalen\n\n"
        
        "3. Ontwerp een gestructureerd leerpad:\n"
        "   - Deel op in logische modules\n"
        "   - Creëer een progressie van eenvoudig naar complex\n"
        "   - Voorzie voorbeelden en oefeningen per module\n\n"
        
        "4. Selecteer geschikte leermiddelen:\n"
        "   - Theoretische uitleg\n"
        "   - Visuele hulpmiddelen\n"
        "   - Praktische opdrachten\n"
        "   - Zelfevaluatiemogelijkheden"
    ),
    expected_output=(
        "Een uitgebreid leerplan document met:\n"
        "- Analyse van het onderwerp en leerdoelen\n"
        "- Gestructureerd leerpad met modules\n"
        "- Geschikte leermethoden per onderdeel\n"
        "- Aanbevolen leermiddelen en oefeningen\n"
        "- Tijdsinschatting per module"
    ),
    agent=planner,
)


# ### Task: Write

# In[62]:


maak_les = Task(
    description=(
        "Maak een les over {onderwerp} met:\n"
        "1. Pakkende intro met leerdoelen\n"
        "2. Theorie uitgelegd via 3 methoden (visueel/verbaal/praktisch)\n"
        "3. Minimaal 2 real-world voorbeelden\n"
        "4. Oefeningen met oplossingssleutel"
    ),
    expected_output="Completeles in Markdown met:\n"
                   "- Interactieve elementen\n"
                   "- Gemarkeerde key takeaways\n"
                   "- Zelfstudie-opdrachten",
    agent=uitlegger
)


# ### Task: Edit

# In[63]:


review_les = Task(
    description=(
        "Review de les over {onderwerp} op:\n"
        "1. Wetenschappelijke nauwkeurigheid\n"
        "2. Didactische effectiviteit\n"
        "3. Praktische toepasbaarheid\n"
        "4. Voeg formatieve evaluatiepunten toe"
    ),
    expected_output="Gereviewde les met:\n"
                   "- Correcties\n"
                   "- Verbetersuggesties\n"
                   "- Extra checkpoints",
    agent=leercoach
)


# ## Creating the Crew
# 
# - Create your crew of Agents
# - Pass the tasks to be performed by those agents.
#     - **Note**: *For this simple example*, the tasks will be performed sequentially (i.e they are dependent on each other), so the _order_ of the task in the list _matters_.
# - `verbose=2` allows you to see all the logs of the execution. 

# In[64]:


crew = Crew(
    agents=[planner, uitlegger, leercoach],
    tasks=[leerplan_taak, maak_les, review_les],
    verbose=2
)


# ## Running the Crew

# **Note**: LLMs can provide different outputs for they same input, so what you get might be different than what you see in the video.

# In[65]:


result = crew.kickoff(inputs={'onderwerp': 'Hoe werkt partieel differentieren?'}
)


# - Display the results of your execution as markdown in the notebook.

# In[66]:


from IPython.display import Markdown
Markdown(result)


# ## Try it Yourself
# 
# - Pass in a topic of your choice and see what the agents come up with!

# In[67]:


topic = "Reinforcement learning"
result = crew.kickoff(inputs={"onderwerp": topic})


# In[68]:


Markdown(result)


# <a name='1'></a>
#  ## Other Popular Models as LLM for your Agents

# #### Hugging Face (HuggingFaceHub endpoint)
# 
# ```Python
# from langchain_community.llms import HuggingFaceHub
# 
# llm = HuggingFaceHub(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     huggingfacehub_api_token="<HF_TOKEN_HERE>",
#     task="text-generation",
# )
# 
# ### you will pass "llm" to your agent function
# ```

# #### Mistral API
# 
# ```Python
# OPENAI_API_KEY=your-mistral-api-key
# OPENAI_API_BASE=https://api.mistral.ai/v1
# OPENAI_MODEL_NAME="mistral-small"
# ```

# #### Cohere
# 
# ```Python
# from langchain_community.chat_models import ChatCohere
# # Initialize language model
# os.environ["COHERE_API_KEY"] = "your-cohere-api-key"
# llm = ChatCohere()
# 
# ### you will pass "llm" to your agent function
# ```

# ### For using Llama locally with Ollama and more, checkout the crewAI documentation on [Connecting to any LLM](https://docs.crewai.com/how-to/LLM-Connections/).

# In[ ]:





# In[ ]:





# In[ ]:




