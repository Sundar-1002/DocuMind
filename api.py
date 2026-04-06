from ingest import ingest
from agent import run_agent
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from strawberry.file_uploads import Upload
import strawberry
import os


@strawberry.type
class Message:
    role: str
    content: str

@strawberry.input
class MessageInput:
    role: str
    content: str

@strawberry.type
class Query:
    @strawberry.field
    def health(self) -> str:
        return "API is healthy and running!"
    
@strawberry.type
class Mutation:
    @strawberry.mutation
    def ask(self, question: MessageInput) -> Message:
        answer = run_agent(question.content)
        return Message(role="assistant", content=answer)

    @strawberry.mutation
    async def upload_documents(self, files: list[Upload]) -> str:
        os.makedirs("data", exist_ok=True)
        for file in files:
            contents = await file.read()
            with open(os.path.join("data", file.filename), "wb") as f:
                f.write(contents)
        ingest()
        return "Documents ingested successfully!"

schema = strawberry.Schema(query = Query, mutation = Mutation)
app = FastAPI()
graphql = GraphQLRouter(schema)
app.include_router(graphql, prefix="/graphql")