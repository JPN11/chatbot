from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pre-trained model and tokenizer
model_name = "microsoft/deberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize chat history
chat_history_ids = None

# Chatbot loop
while True:
    # Get user input
    user_input = input(">> User: ")
    
    # Tokenize user input and append to chat history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Create attention mask
    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id, 
        attention_mask=attention_mask
    )
    
    # Decode and print response
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Bot:", bot_response)