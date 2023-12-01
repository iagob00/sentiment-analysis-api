from peewee import Model, SqliteDatabase, TextField

db =SqliteDatabase('sentiments.db')

class SentimentModel(Model):
    text = TextField()
    sentiment = TextField(null = True)

    class Meta(): 
        database = db

db.create_tables([SentimentModel])  