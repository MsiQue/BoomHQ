import tqdm
from transformers import AutoTokenizer, AutoModel
from global_info import get_database_cursor

def get_embedding(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
    return cls_embedding.tolist()

def add_vector_column(cursor, tokenizer, model, table_name, column_name):
    vector_column_name = f"{column_name}_vector"
    vector_dim = 768
    sql = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {vector_column_name} vector({vector_dim})"
    cursor.execute(sql)

    select_query = f"SELECT p_partkey, {column_name} FROM {table_name}"
    cursor.execute(select_query)
    rows = cursor.fetchall()
    for row in tqdm.tqdm(rows):
        p_partkey_value = row[0]
        text = row[1]
        embedding = get_embedding(tokenizer, model, text)
        if embedding is not None:
            update_query = f"UPDATE {table_name} SET {vector_column_name} = %s WHERE p_partkey = %s"
            cursor.execute(update_query, (embedding, p_partkey_value))

if __name__ == '__main__':
    cursor = get_database_cursor('tpch')

    LM_path = 'LM/bert'
    tokenizer = AutoTokenizer.from_pretrained(LM_path)
    model = AutoModel.from_pretrained(LM_path)

    add_vector_column(cursor, tokenizer, model, 'part', 'p_comment')