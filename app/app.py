import streamlit as st
import random
import time
import torch
from transformers import (
      BertTokenizerFast,
      BertForSequenceClassification,
      T5Tokenizer,
      T5ForConditionalGeneration
)
import re


def model_generation(prompt):


  # we load the classifier model we trained earlier
  classifier_model_path = './app/science_classifier_model'
  classifier_tokenizer = BertTokenizerFast.from_pretrained('./app/TOKEIZERscience_classifier_model')
  classifier_model = BertForSequenceClassification.from_pretrained(classifier_model_path)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  classifier_model.to(device)
  classifier_model.eval()

  # we will load a generative model for answer generation
  generator_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
  generator_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
  generator_model.to(device)
  generator_model.eval()

  # we will list science keywords to use them as an extra layer of keyword based classification
  science_keywords = [
    "gravity", "atom", "molecule", "photosynthesis", "evolution", "energy", "force",
    "planet", "star", "chemical", "biology", "physics", "matter", "ecosystem",
    "light", "solar system", "electricity", "magnetism", "temperature", "velocity",
    "acceleration", "mass", "nucleus", "protein", "cell","force", "energy", "gravity", "velocity", "acceleration", "mass", "momentum",
    "inertia", "friction", "magnetism", "electricity", "voltage", "resistance",
    "current", "circuit", "kinetic", "potential energy", "thermodynamics", "heat",
    "temperature", "radiation", "quantum", "wave", "frequency", "optics", "light",
    "refraction", "reflection", "laser", "sound", "atom", "particle", "electron",
    "proton", "neutron", "nucleus", "fusion", "fission", "relativity", "force field",
    "entropy", "molecule", "compound", "element", "chemical", "reaction", "bond",
    "ionic", "covalent", "acid", "base", "pH", "isotope", "catalyst", "organic",
    "inorganic", "periodic table", "oxidation", "polymer", "solution", "solvent",
    "solute", "concentration", "equilibrium", "gas", "liquid", "solid", "combustion",
    "distillation", "crystallization", "hydrocarbon", "alkali", "electrolyte",
    "metal", "nonmetal", "cell", "DNA", "gene", "chromosome", "photosynthesis",
    "respiration", "bacteria", "virus", "evolution", "adaptation", "species",
    "ecosystem", "biodiversity", "chloroplast", "enzyme", "protein", "carbohydrate",
    "lipid", "organ", "tissue", "hormone", "neuron", "ecology", "population",
    "organism", "metabolism", "mutation", "genetics", "inheritance", "biochemistry",
    "zoology", "botany", "microbiology", "vaccine", "antibody", "geology",
    "tectonic", "earthquake", "volcano", "sediment", "erosion", "fossil",
    "mineral", "rock", "magma", "crust", "mantle", "core", "atmosphere",
    "climate", "weather", "storm", "hurricane", "tornado", "precipitation",
    "cloud", "water cycle", "ocean", "glacier", "tsunami", "flood", "drought",
    "meteorology", "hydrology", "geosphere", "lithosphere", "stratosphere",
    "troposphere", "ozone layer", "pollution", "carbon cycle", "nitrogen cycle",
    "universe", "galaxy", "star", "planet", "asteroid", "comet", "orbit",
    "black hole", "nebula", "supernova", "solar system", "moon", "sun", "light-year",
    "spacecraft", "astronaut", "telescope", "cosmic", "asteroid belt", "meteor",
    "satellite", "mars", "jupiter", "saturn", "neptune", "uranus", "mercury",
    "venus", "milky way", "big bang", "dark matter", "dark energy", "exoplanet",
    "red giant", "white dwarf", "climate change", "global warming", "greenhouse effect",
    "carbon dioxide", "renewable energy", "fossil fuel", "biodiversity", "conservation",
    "sustainability", "pollution", "deforestation", "recycling", "ozone layer",
    "acid rain", "carbon footprint", "solar energy", "wind energy", "hydropower",
    "sustainability", "environmental impact", "organic farming", "endangered species",
    "habitat destruction", "water pollution", "air quality", "experiment",
    "hypothesis", "theory", "observation", "data", "research", "analysis",
    "measurement", "discovery", "technology", "invention", "laboratory",
    "scientist", "equation", "variable", "constant", "simulation", "model",
    "result", "conclusion", "method", "investigation", "microscope", "telescope",
    "biology", "chemistry", "physics", "ecosystem", "organ", "heart", "brain",
    "lung", "kidney", "liver", "stomach", "muscle", "bone", "skeleton",
    "skin", "tissue", "neuron", "blood", "circulatory system", "respiratory system",
    "digestive system", "nervous system", "immune system", "endocrine system",
    "hormone", "reproductive system", "DNA", "gene", "chromosome", "enzyme",
    "antibody", "vaccine", "algebra", "equation", "formula", "geometry",
    "calculus", "trigonometry", "fraction", "probability", "statistics",
    "mean", "median", "mode", "graph", "measurement", "vector", "matrix",
    "ratio", "derivative", "integral", "function", "slope", "variable",
    "theorem", "exponent", "algorithm", "circumference", "diameter", "radius",
    "pi (π)"

        # we can add more keywords as we need to help with more classification
    ]

  # Function to check if the question contains science related keywords
  def contains_science_keyword(question):
      question = question.lower()
      for keyword in science_keywords:
          if keyword in question:
              return True
      return False
  
  # we create a function to classify the question

  def is_science_question(question):
      # Step 1: Use keyword-based classifier
      if contains_science_keyword(question):
        return True  # now if the question is scince related it will classify it

      # we now utilize the trained transformer based classifier
      inputs = classifier_tokenizer(
          question,
          return_tensors='pt',
          truncation=True,
          padding=True,
          max_length=128
      ).to(device)

      with torch.no_grad():
          outputs = classifier_model(**inputs)
      logits = outputs.logits
      probabilities = torch.softmax(logits, dim=1)
      predicted_label = torch.argmax(probabilities, dim=1).item()

      # label 1 is for science and 0 for non science as we labeled the data earlier
      return predicted_label == 1
  
  # List of greetings in case the user greeted the bot
  greetings = ["hello", "hi", "hey", "how are you", "good morning", "good evening", "what's up", "hola", "hay", "hie","hoi"]

  def is_greeting(user_input):
    # Normalize the input to lowercase and check if it's a greeting
    user_input = user_input.lower().strip()
    return any(greet in user_input for greet in greetings)
  
  exits = ["exit","see you later","see you","bye","goodbye","good bye"]

  def end_chat(user_input):
    user_input = user_input.lower().strip()
    return any(end in user_input for end in exits)

  # a function to clean the generated answer by removing unwanted characters and html tags
  def clean_answer(answer):
    # removing any occurrences of multiple parentheses
    answer = re.sub(r'\)+', '', answer)
    # remove any HTML tags like <br>, </br>, etc.
    answer = re.sub(r'<.*?>', '', answer)
    # triiming  any extra whitespace
    return answer.strip()
  
  # the function to generate response
  def generate_answer(question):
      # if the input is a greeting
      if is_greeting(question):
        return "Hello! How can I help you with your science questions today?"
      
      # if the inout is exit
      if end_chat(question):
          return "Goodbye!"

      # if it's not a greeting, proceed with science question classification
      if not is_science_question(question):
          return "I'm sorry, but I can only answer science questions for kids. Please ask me something related to science!"

      # the input that will go to the generative model
      input_text = (
          f"You are a friendly and knowledgeable science teacher explaining concepts to kids. "
          f"Provide a detailed and engaging answer to the following science question.\n\n"
          f"Question: {question}\n\n"
          f"Answer:"
      )

      # we tokenize the input
      inputs = generator_tokenizer.encode(
          input_text,
          return_tensors='pt',
          max_length=512,
          truncation=True
      ).to(device)

      # then generate the answer
      with torch.no_grad():
          outputs = generator_model.generate(
              inputs,
              do_sample=True,
              max_length=300,
              min_length=30,
              temperature=0.5,
              top_p=0.9,
              repetition_penalty=1.2,
              no_repeat_ngram_size=3,
          )

      # decoding the answer before passing it
      answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

      # cleaning the answer using our cleaning function
      answer = clean_answer(answer)

      # we remove the prompt from the response that will go to the user
      if answer.startswith(input_text):
          answer = answer[len(input_text):].strip()

      return answer.strip()
  return generate_answer(prompt)

def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
st.title("Science Bot for Kids")
st.write("Ask science-related questions and get answers designed for kids.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # need to add contition to exit

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        #response = st.write_stream(generate_answer(prompt))
        #response = st.write(model_generation(prompt))
        response = model_generation(prompt)
        st.markdown(response)


    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
