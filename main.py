from fastapi import FastAPI

app = FastAPI()


@app.post('/initialize')
async def initialize():
    return {'message': 'Initialized'}

