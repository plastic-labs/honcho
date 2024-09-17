import logging
import uuid
import re
from rich import print as rprint
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models
from .voe import tom_inference, user_representation
from .timing import timing_decorator, csv_file_path

# Turn off SQLAlchemy Echo logging
logging.getLogger("sqlalchemy.engine.Engine").disabled = True


# FIXME see if this is SAFE
async def add_metamessage(db, message_id, metamessage_type, content):
    metamessage = models.Metamessage(
        message_id=message_id,
        metamessage_type=metamessage_type,
        content=content,
        h_metadata={},
    )
    db.add(metamessage)

def parse_xml_content(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

async def process_item(db: AsyncSession, payload: dict, enable_timing: bool = False):
    processing_args = [
        payload["content"],
        payload["app_id"],
        payload["user_id"],
        payload["session_id"],
        payload["message_id"],
        db,
        enable_timing,
    ]
    if payload["is_user"]:
        await process_user_message(*processing_args)
    else:
        await process_ai_message(*processing_args)
    return


@timing_decorator(csv_file_path)
async def process_ai_message(
    content: str,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    db: AsyncSession,
    enable_timing: bool = False,
):
    """
    Process an AI message. Make a prediction about what the user is going to say to it.
    """
    rprint(f"[green]Processing AI message: {content}[/green]")

    subquery = (
        select(models.Message.created_at)
        .where(models.Message.id == message_id)
        .scalar_subquery()
    )
    messages_stmt = (
        select(models.Message)
        .where(models.Message.session_id == session_id)
        .order_by(models.Message.created_at.desc())
        .where(models.Message.created_at < subquery)
        .limit(10)
    )

    result = await db.execute(messages_stmt)
    messages = result.scalars().all()[::-1]

    chat_history_str = "\n".join(
        [f"human: {m.content}" if m.is_user else f"ai: {m.content}" for m in messages]
    )
    # append current message to chat history
    chat_history_str = f"{chat_history_str}\nai: {content}"

<<<<<<< HEAD
    tom_inference_response = await tom_inference(chat_history_str, session_id=session_id, enable_timing=enable_timing)
=======
    # user prediction thought
    user_prediction_thought = UserPredictionThought(chat_history=chat_history_str)
    user_prediction_thought_response = user_prediction_thought.call()
>>>>>>> main

    prediction = parse_xml_content(tom_inference_response, "prediction")

    await add_metamessage(
        db,
        message_id,
        "tom_inference",
        prediction,
    )
<<<<<<< HEAD
=======
    additional_data_set = set()
    additional_data_list = []
    for d in additional_data:
        response = await crud.query_documents(
            db,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
            query=d,
            top_k=3,
        )
        for document in response:
            additional_data_set.add(document)
        additional_data_list = list(additional_data_set)

    context_str = "\n".join([document.content for document in additional_data_list])

    # user prediction thought revision given the context
    user_prediction_thought_revision = UserPredictionThoughtRevision(
        user_prediction_thought=user_prediction_thought_response.content,
        retrieved_context=context_str,
        chat_history=chat_history_str,
    )
    user_prediction_thought_revision_response = user_prediction_thought_revision.call()

    if user_prediction_thought_revision_response.content == "None":
        rprint("[blue]Model predicted no changes to the user prediction thought")
        await add_metamessage(
            db,
            message_id,
            "user_prediction_thought",
            user_prediction_thought_response.content,
        )
        await add_metamessage(
            db,
            message_id,
            "user_prediction_thought_revision",
            user_prediction_thought_response.content,
        )
    else:
        await add_metamessage(
            db,
            message_id,
            "user_prediction_thought",
            user_prediction_thought_response.content,
        )
        await add_metamessage(
            db,
            message_id,
            "user_prediction_thought_revision",
            user_prediction_thought_revision_response.content,
        )
>>>>>>> main

    await db.commit()


    rprint("[blue]Tom Inference:")
    content_lines = str(prediction)
    rprint(f"[blue]{content_lines}")

@timing_decorator(csv_file_path)
async def process_user_message(
    content: str,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    db: AsyncSession,
    enable_timing: bool = False,
):
    """
    Process a user message. If there are revised user predictions to run VoE against, run it. Otherwise pass.
    """
    rprint(f"[orange1]Processing User Message: {content}")

    # Get the AI message directly preceding this User message
    subquery = (
        select(models.Message.created_at)
        .where(models.Message.id == message_id)
        .scalar_subquery()
    )

    messages_stmt = (
        select(models.Message)
        .where(models.Message.session_id == session_id)
        .where(models.Message.is_user == False)
        .order_by(models.Message.created_at.desc())
        .where(models.Message.created_at < subquery)
        .limit(1)
    )

    response = await db.execute(messages_stmt)
    ai_message = response.scalar_one_or_none()

    if ai_message and ai_message.content:
        rprint(f"[orange1]AI Message: {ai_message.content}")
<<<<<<< HEAD
        
        # Fetch the tom_inference metamessage
        tom_inference_stmt = (
=======
        # Get the User Thought Revision Associated with this AI Message
        metamessages_stmt = (
>>>>>>> main
            select(models.Metamessage)
            .where(models.Metamessage.message_id == ai_message.id)
            .where(models.Metamessage.metamessage_type == "tom_inference")
            .order_by(models.Metamessage.created_at.asc())
            .limit(1)
        )
        response = await db.execute(tom_inference_stmt)
        tom_inference_metamessage = response.scalar_one_or_none()

        if tom_inference_metamessage and tom_inference_metamessage.content:
            rprint(f"[orange1]Tom Inference: {tom_inference_metamessage.content}")

            # Fetch the existing user representation
            user_representation_stmt = (
                select(models.Metamessage)
                .where(models.Metamessage.message_id == ai_message.id)
                .where(models.Metamessage.metamessage_type == "user_representation")
                .order_by(models.Metamessage.created_at.desc())
                .limit(1)
            )
<<<<<<< HEAD
            response = await db.execute(user_representation_stmt)
            existing_representation = response.scalar_one_or_none()
=======
            voe_thought_response = voe_thought.call()
>>>>>>> main

            existing_representation_content = existing_representation.content if existing_representation else "None"

            # Call user_representation
            user_representation_response = await user_representation(
                chat_history=f"{ai_message.content}\nhuman: {content}",
                session_id=session_id,
                user_representation=existing_representation_content,
                tom_inference=tom_inference_metamessage.content,
                enable_timing=enable_timing
            )
<<<<<<< HEAD
=======
            voe_derive_facts_response = voe_derive_facts.call()
>>>>>>> main

            # Store the user_representation response as a metamessage
            await add_metamessage(
                db,
                message_id,
                "user_representation",
                user_representation_response,
            )

            rprint("[orange1]User Representation:")
            rprint(f"[orange1]{user_representation_response}")

        else:
<<<<<<< HEAD
            raise Exception("\033[91mTom Inference NOT READY YET")
    else:
        rprint("[red]No AI message before this user message[/red]")
        return
=======
            rprint("[red] No Prediction Associated with this Message")
            return
    else:
        rprint("[red]No AI message before this user message[/red]")
        return


async def check_dups(
    app_id: uuid.UUID, user_id: uuid.UUID, collection_id: uuid.UUID, facts: List[str]
):
    """Check that we're not storing duplicate facts"""

    check_duplication = CheckVoeList(existing_facts=[], new_fact="")
    result = None
    new_facts = []
    # global_existing_facts = []  # for debugging
    for fact in facts:
        async with SessionLocal() as db:
            result = await crud.query_documents(
                db=db,
                app_id=app_id,
                user_id=user_id,
                collection_id=collection_id,
                query=fact,
                top_k=5,
            )
        existing_facts = [document.content for document in result]
        if len(existing_facts) == 0:
            new_facts.append(fact)
            rprint(f"[light_steel_blue]New Fact: {fact}")
            continue

        # global_existing_facts.extend(existing_facts)  # for debugging

        check_duplication.existing_facts = existing_facts
        check_duplication.new_fact = fact
        response = check_duplication.call()
        rprint("[light_steel_blue]==================")
        rprint(f"[light_steel_blue]Dedupe Responses: {response.content}")
        rprint("[light_steel_blue]==================")
        if response.content == "true":
            new_facts.append(fact)
            rprint(f"[light_steel_blue]New Fact: {fact}")
            continue

    rprint("[light_steel_blue]===================")
    # rprint("[light_steel_blue]Existing Facts:")
    # rprint(global_existing_facts)
    rprint("[light_steel_blue]Net New Facts:")
    rprint(new_facts)
    rprint("[light_steel_blue]===================")
    return new_facts
>>>>>>> main
