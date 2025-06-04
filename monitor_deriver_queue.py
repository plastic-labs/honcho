#!/usr/bin/env python3
"""
Monitor the Honcho deriver queue and show progress as messages are processed.
This is a read-only monitor that does not modify the queue in any way.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path so we can import Honcho modules
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv

# Import Honcho's actual models and dependencies
from src import models
from src.dependencies import tracked_db

load_dotenv()


async def get_queue_stats(db: AsyncSession):
    """Get current queue statistics (read-only)"""
    # Count unprocessed messages
    unprocessed_result = await db.execute(
        select(func.count()).select_from(models.QueueItem).where(models.QueueItem.processed == False)
    )
    unprocessed_count = unprocessed_result.scalar() or 0
    
    # Count total messages
    total_result = await db.execute(
        select(func.count()).select_from(models.QueueItem)
    )
    total_count = total_result.scalar() or 0
    
    # Count messages by session
    session_stats_result = await db.execute(
        select(models.QueueItem.session_id, func.count())
        .where(models.QueueItem.processed == False)
        .group_by(models.QueueItem.session_id)
    )
    session_stats = list(session_stats_result)
    sessions_with_messages = len(session_stats)
    
    # Get active processing sessions
    active_sessions_result = await db.execute(
        select(func.count()).select_from(models.ActiveQueueSession)
    )
    active_processing = active_sessions_result.scalar() or 0
    
    return unprocessed_count, total_count, sessions_with_messages, active_processing


async def get_session_details(db: AsyncSession):
    """Get details about sessions in the queue (read-only)"""
    result = await db.execute(
        select(
            models.QueueItem.session_id,
            func.count(models.QueueItem.id).label('message_count'),
            func.sum(case((models.QueueItem.processed == True, 1), else_=0)).label('processed_count')
        )
        .group_by(models.QueueItem.session_id)
        .order_by(func.count(models.QueueItem.id).desc())
        .limit(10)
    )
    return result.all()


async def monitor_queue(poll_interval=2.0):
    """Monitor the queue and display progress (read-only)"""
    try:
        print("🔍 Honcho Deriver Queue Monitor")
        print("================================\n")
        print("Connecting to database...")
        
        # Use Honcho's tracked_db for read-only access
        async with tracked_db("queue_monitor_readonly") as db:
            # Get initial stats
            unprocessed, total, sessions_with_msgs, active_processing = await get_queue_stats(db)
            
            if total == 0:
                print("📭 Queue is empty! No messages in the queue.")
                return
            
            if unprocessed == 0:
                print("✅ All messages have been processed!")
                print(f"   Total processed: {total}")
                return
            
            print(f"\n📊 Initial Queue Status:")
            print(f"  • Total messages: {total:,}")
            print(f"  • Unprocessed: {unprocessed:,}")
            print(f"  • Processed: {total - unprocessed:,}")
            print(f"  • Sessions with messages: {sessions_with_msgs}")
            print(f"  • Active processing sessions: {active_processing}")
            
            # Show top sessions
            print(f"\n📋 Top Sessions by Message Count:")
            session_details = await get_session_details(db)
            for session_id, msg_count, processed in session_details:
                unproc = msg_count - (processed or 0)
                print(f"  • Session {session_id}: {msg_count} messages ({unproc} unprocessed)")
            
            print(f"\n⏳ Monitoring queue (polling every {poll_interval}s)...")
            print("   Press Ctrl+C to stop\n")
        
        # Create progress bar
        with tqdm(total=unprocessed, desc="Processing messages", unit="msg", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            initial_unprocessed = unprocessed
            last_unprocessed = unprocessed
            start_time = asyncio.get_event_loop().time()
            
            while unprocessed > 0:
                await asyncio.sleep(poll_interval)
                
                # Get updated stats (read-only)
                async with tracked_db("queue_monitor_readonly") as db:
                    unprocessed, total, sessions_with_msgs, active_processing = await get_queue_stats(db)
                
                # Update progress
                processed_since_last = last_unprocessed - unprocessed
                if processed_since_last > 0:
                    pbar.update(processed_since_last)
                    last_unprocessed = unprocessed
                elif processed_since_last < 0:
                    # Handle case where new messages were added
                    print(f"\n⚠️  {-processed_since_last} new messages added to queue")
                    pbar.total = pbar.n + unprocessed
                    pbar.refresh()
                    last_unprocessed = unprocessed
                
                # Calculate rate
                elapsed = asyncio.get_event_loop().time() - start_time
                processed_total = initial_unprocessed - unprocessed
                rate = processed_total / elapsed if elapsed > 0 else 0
                
                # Update description with current stats
                pbar.set_description(
                    f"Processing (Active: {active_processing}, Sessions: {sessions_with_msgs})"
                )
                
                # Show additional stats
                pbar.set_postfix_str(
                    f"Remaining: {unprocessed:,} | Rate: {rate:.1f} msg/s"
                )
        
        print(f"\n✅ Queue processing complete!")
        print(f"   • Total processed: {initial_unprocessed:,} messages")
        print(f"   • Time taken: {elapsed:.1f} seconds")
        print(f"   • Average rate: {rate:.1f} messages/second")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main entry point"""
    # Check if we can connect to the database
    try:
        async with tracked_db("queue_monitor_test") as db:
            # Just test the connection
            await db.execute(select(1))
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        print("\nMake sure:")
        print("1. PostgreSQL is running")
        print("2. CONNECTION_URI environment variable is set correctly")
        print("3. The database and tables exist (run migrations if needed)")
        return
    
    # Run the monitor
    await monitor_queue(poll_interval=2.0)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 