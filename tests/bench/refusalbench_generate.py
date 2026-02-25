"""
RefusalBench Dataset Generation Script

Generates a full RefusalBench dataset using OpenRouter LLMs to create perturbations
from seed QA pairs. Produces JSONL output compatible with refusalbench.py and
refusalbench_baseline.py runners.

Pipeline:
    1. Generate: For each seed x (class, intensity), call LLM with catalogue prompt
    2. Verify (optional): Call verifier LLM, filter to PASS only
    3. Write: Output surviving entries to JSONL

Usage:
    uv run python -m tests.bench.refusalbench_generate
    uv run python -m tests.bench.refusalbench_generate --verify
    uv run python -m tests.bench.refusalbench_generate --perturbation-class P-Ambiguity --intensity LOW

Reference: arXiv:2510.10390
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .refusalbench_catalogue import RefusalBenchCatalogue

# Load .env from bench directory
bench_dir = Path(__file__).parent
load_dotenv(bench_dir / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SeedQA:
    """A seed question-answer pair for perturbation generation."""

    qid: str
    query: str
    answer: list[str]
    context: str


@dataclass
class GeneratedEntry:
    """A single generated RefusalBench entry."""

    unique_id: str
    perturbation_class: str
    intensity: str
    original_query: str
    perturbed_query: str
    original_context: str
    perturbed_context: str
    original_answers: list[str]
    expected_rag_behavior: str
    lever_selected: str = ""
    implementation_reasoning: str = ""
    generation_model: str = ""

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Convert to the RefusalBenchEntry-compatible dict for JSONL output."""
        return {
            "unique_id": self.unique_id,
            "perturbation_class": self.perturbation_class,
            "intensity": self.intensity,
            "original_query": self.original_query,
            "perturbed_query": self.perturbed_query,
            "original_context": self.original_context,
            "perturbed_context": self.perturbed_context,
            "original_answers": self.original_answers,
            "expected_rag_behavior": self.expected_rag_behavior,
        }

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert to a full metadata dict including generation details."""
        d = self.to_jsonl_dict()
        d["lever_selected"] = self.lever_selected
        d["implementation_reasoning"] = self.implementation_reasoning
        d["generation_model"] = self.generation_model
        return d


@dataclass
class GenerationStats:
    """Statistics for the generation pipeline."""

    total_attempted: int = 0
    generation_success: int = 0
    generation_failed: int = 0
    verification_pass: int = 0
    verification_fail: int = 0
    verification_error: int = 0
    final_count: int = 0
    start_time: float = field(default_factory=time.time)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict."""
        elapsed = time.time() - self.start_time
        return {
            "total_attempted": self.total_attempted,
            "generation_success": self.generation_success,
            "generation_failed": self.generation_failed,
            "verification_pass": self.verification_pass,
            "verification_fail": self.verification_fail,
            "verification_error": self.verification_error,
            "final_count": self.final_count,
            "elapsed_seconds": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# Built-in seed data (100 diverse QA pairs)
# ---------------------------------------------------------------------------

BUILTIN_SEEDS: list[dict[str, Any]] = [
    # Science
    {"qid": "sci_001", "query": "What is the speed of light in a vacuum?", "answer": ["299,792,458 meters per second", "approximately 300,000 km/s"], "context": "The speed of light in a vacuum is exactly 299,792,458 meters per second (approximately 300,000 km/s or 186,000 miles per second). This is a fundamental physical constant denoted by the letter c."},
    {"qid": "sci_002", "query": "What is the chemical formula for water?", "answer": ["H2O"], "context": "Water is a chemical substance with the formula H2O. Each molecule consists of two hydrogen atoms covalently bonded to one oxygen atom. Water covers about 71% of the Earth's surface."},
    {"qid": "sci_003", "query": "What is the boiling point of water at sea level?", "answer": ["100 degrees Celsius", "212 degrees Fahrenheit"], "context": "At standard atmospheric pressure (sea level), water boils at 100 degrees Celsius (212 degrees Fahrenheit). The boiling point decreases at higher altitudes due to lower atmospheric pressure."},
    {"qid": "sci_004", "query": "How many chromosomes do humans have?", "answer": ["46", "23 pairs"], "context": "Humans typically have 46 chromosomes, arranged in 23 pairs. Of these, 22 pairs are autosomes, and one pair consists of sex chromosomes (XX for females, XY for males)."},
    {"qid": "sci_005", "query": "What is the most abundant gas in Earth's atmosphere?", "answer": ["nitrogen", "N2"], "context": "Nitrogen (N2) is the most abundant gas in Earth's atmosphere, comprising approximately 78.09% by volume. Oxygen is second at about 20.95%, followed by argon at 0.93%."},
    {"qid": "sci_006", "query": "What is the atomic number of carbon?", "answer": ["6"], "context": "Carbon is a chemical element with the symbol C and atomic number 6. It is the 15th most abundant element in the Earth's crust and the fourth most abundant element in the universe by mass."},
    {"qid": "sci_007", "query": "What is the largest organ in the human body?", "answer": ["skin"], "context": "The skin is the largest organ of the human body, with a total area of about 20 square feet (1.85 square meters) in adults. It serves as a protective barrier and helps regulate body temperature."},
    {"qid": "sci_008", "query": "What causes tides on Earth?", "answer": ["gravitational pull of the Moon and Sun", "Moon's gravity"], "context": "Tides are caused primarily by the gravitational pull of the Moon on Earth's oceans. The Sun also contributes, but its effect is about 46% that of the Moon due to its much greater distance despite its larger mass."},
    {"qid": "sci_009", "query": "What is photosynthesis?", "answer": ["process by which plants convert light energy into chemical energy"], "context": "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. The overall equation is 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2."},
    {"qid": "sci_010", "query": "What is the pH of pure water?", "answer": ["7"], "context": "Pure water has a pH of 7, which is considered neutral. The pH scale ranges from 0 to 14, with values below 7 being acidic and values above 7 being basic (alkaline)."},
    # History
    {"qid": "hist_001", "query": "When did World War II end?", "answer": ["1945", "September 2, 1945"], "context": "World War II ended on September 2, 1945, when Japan formally surrendered aboard the USS Missouri in Tokyo Bay. The war in Europe had ended on May 8, 1945 (V-E Day) with Germany's unconditional surrender."},
    {"qid": "hist_002", "query": "Who was the first President of the United States?", "answer": ["George Washington"], "context": "George Washington served as the first President of the United States from 1789 to 1797. He was unanimously elected by the Electoral College and is often referred to as the 'Father of His Country.'"},
    {"qid": "hist_003", "query": "When was the Declaration of Independence signed?", "answer": ["1776", "August 2, 1776"], "context": "The United States Declaration of Independence was adopted by the Continental Congress on July 4, 1776. The actual signing by most delegates took place on August 2, 1776."},
    {"qid": "hist_004", "query": "What year did the Berlin Wall fall?", "answer": ["1989"], "context": "The Berlin Wall fell on November 9, 1989, when the East German government opened the border crossings. The wall had divided Berlin since August 13, 1961, and its fall symbolized the end of the Cold War."},
    {"qid": "hist_005", "query": "Who invented the printing press?", "answer": ["Johannes Gutenberg"], "context": "Johannes Gutenberg invented the movable-type printing press around 1440 in Mainz, Germany. His invention revolutionized book production and is considered one of the most important inventions in human history."},
    {"qid": "hist_006", "query": "When did the French Revolution begin?", "answer": ["1789"], "context": "The French Revolution began in 1789 with the storming of the Bastille on July 14. It lasted until 1799 and resulted in the overthrow of the monarchy and establishment of a republic."},
    {"qid": "hist_007", "query": "Who built the Great Wall of China?", "answer": ["multiple Chinese dynasties", "Qin Dynasty initiated it"], "context": "The Great Wall of China was built over many centuries by multiple Chinese dynasties. Construction began under Emperor Qin Shi Huang around 221 BC, but most of the existing wall was built during the Ming Dynasty (1368-1644)."},
    {"qid": "hist_008", "query": "When was the Magna Carta signed?", "answer": ["1215"], "context": "The Magna Carta was sealed by King John of England on June 15, 1215, at Runnymede. It established the principle that everyone, including the king, was subject to the law."},
    {"qid": "hist_009", "query": "Who was the first person to walk on the Moon?", "answer": ["Neil Armstrong"], "context": "Neil Armstrong became the first person to walk on the Moon on July 20, 1969, during the Apollo 11 mission. His famous words were 'That's one small step for man, one giant leap for mankind.'"},
    {"qid": "hist_010", "query": "What ancient civilization built the pyramids at Giza?", "answer": ["ancient Egyptians", "Egypt"], "context": "The pyramids at Giza were built by the ancient Egyptians during the Old Kingdom period, approximately 2580-2560 BC. The Great Pyramid was built for Pharaoh Khufu and is the oldest of the Seven Wonders of the Ancient World."},
    # Geography
    {"qid": "geo_001", "query": "What is the largest country by area?", "answer": ["Russia"], "context": "Russia is the largest country in the world by area, spanning 17,098,242 square kilometers (6,601,670 square miles). It extends across eleven time zones and shares borders with sixteen countries."},
    {"qid": "geo_002", "query": "What is the longest river in the world?", "answer": ["Nile", "Amazon"], "context": "The Nile River is traditionally considered the longest river at approximately 6,650 km (4,130 miles), though some measurements suggest the Amazon may be longer at about 6,400 km. The debate continues among geographers."},
    {"qid": "geo_003", "query": "What is the deepest ocean trench?", "answer": ["Mariana Trench"], "context": "The Mariana Trench in the western Pacific Ocean is the deepest known oceanic trench. Its deepest point, Challenger Deep, reaches approximately 10,994 meters (36,070 feet) below sea level."},
    {"qid": "geo_004", "query": "How many continents are there?", "answer": ["7"], "context": "There are seven continents on Earth: Africa, Antarctica, Asia, Australia/Oceania, Europe, North America, and South America. Asia is the largest by both area and population."},
    {"qid": "geo_005", "query": "What is the capital of Japan?", "answer": ["Tokyo"], "context": "Tokyo is the capital city of Japan. It is located on the southeastern coast of Honshu, the largest of Japan's main islands. The Greater Tokyo Area is the most populous metropolitan area in the world with over 37 million people."},
    {"qid": "geo_006", "query": "What is the highest mountain in the world?", "answer": ["Mount Everest"], "context": "Mount Everest is the highest mountain in the world, standing at 8,849 meters (29,032 feet) above sea level. It is located in the Mahalangur Himal sub-range of the Himalayas on the border between Nepal and Tibet."},
    {"qid": "geo_007", "query": "What is the largest desert in the world?", "answer": ["Antarctic Desert", "Sahara"], "context": "The Antarctic Desert is technically the largest desert in the world at 14.2 million square kilometers. The Sahara is the largest hot desert at 9.2 million square kilometers, covering much of North Africa."},
    {"qid": "geo_008", "query": "Which country has the most islands?", "answer": ["Sweden"], "context": "Sweden has the most islands in the world with approximately 267,570 islands. Indonesia is second with about 17,508 islands, and Finland third with around 178,947 islands."},
    {"qid": "geo_009", "query": "What is the smallest country in the world?", "answer": ["Vatican City"], "context": "Vatican City is the smallest country in the world by both area (0.44 square kilometers) and population (about 800 people). It is an independent city-state enclaved within Rome, Italy."},
    {"qid": "geo_010", "query": "What ocean lies between Africa and Australia?", "answer": ["Indian Ocean"], "context": "The Indian Ocean lies between Africa to the west and Australia to the east. It is the third-largest ocean, covering approximately 70.56 million square kilometers (27.24 million square miles)."},
    # Technology
    {"qid": "tech_001", "query": "Who founded Apple Inc.?", "answer": ["Steve Jobs, Steve Wozniak, and Ronald Wayne"], "context": "Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne. The company started in the Jobs family garage in Los Altos, California, building and selling hand-built computers."},
    {"qid": "tech_002", "query": "What does CPU stand for?", "answer": ["Central Processing Unit"], "context": "CPU stands for Central Processing Unit. It is the primary component of a computer that performs most of the processing inside the computer. Modern CPUs contain billions of transistors."},
    {"qid": "tech_003", "query": "When was the World Wide Web invented?", "answer": ["1989", "1990"], "context": "The World Wide Web was invented by Tim Berners-Lee in 1989 while working at CERN. He wrote the first web browser in 1990. The first website went live on August 6, 1991."},
    {"qid": "tech_004", "query": "What programming language was created by Guido van Rossum?", "answer": ["Python"], "context": "Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability and supports multiple programming paradigms including procedural, object-oriented, and functional programming."},
    {"qid": "tech_005", "query": "What does HTML stand for?", "answer": ["HyperText Markup Language"], "context": "HTML stands for HyperText Markup Language. It is the standard markup language for documents designed to be displayed in a web browser. HTML elements are the building blocks of HTML pages."},
    {"qid": "tech_006", "query": "When was the first iPhone released?", "answer": ["2007", "June 29, 2007"], "context": "The first iPhone was released on June 29, 2007. It was announced by Steve Jobs at the Macworld Conference on January 9, 2007. The original iPhone had a 3.5-inch display and 2-megapixel camera."},
    {"qid": "tech_007", "query": "What company developed the Android operating system?", "answer": ["Google", "Android Inc."], "context": "Android was originally developed by Android Inc., which was founded by Andy Rubin in 2003. Google acquired Android Inc. in 2005 and released the first commercial Android device in 2008."},
    {"qid": "tech_008", "query": "What is the largest social media platform by users?", "answer": ["Facebook", "Meta"], "context": "Facebook, owned by Meta Platforms, is the largest social media platform with approximately 3 billion monthly active users as of 2024. It was founded by Mark Zuckerberg in 2004 at Harvard University."},
    {"qid": "tech_009", "query": "What does USB stand for?", "answer": ["Universal Serial Bus"], "context": "USB stands for Universal Serial Bus. It is an industry standard that establishes specifications for cables, connectors, and protocols for connection and communication between computers and devices."},
    {"qid": "tech_010", "query": "Who is the CEO of Tesla?", "answer": ["Elon Musk"], "context": "Elon Musk is the CEO of Tesla, Inc. He joined the company in 2004 as chairman of the board after leading its Series A financing round. He became CEO in 2008 and has led the company since."},
    # Medicine
    {"qid": "med_001", "query": "What is the normal human body temperature?", "answer": ["37 degrees Celsius", "98.6 degrees Fahrenheit"], "context": "The normal human body temperature is approximately 37 degrees Celsius (98.6 degrees Fahrenheit), though it can vary slightly between individuals and throughout the day. Fever is generally defined as a temperature above 38 C (100.4 F)."},
    {"qid": "med_002", "query": "Who discovered penicillin?", "answer": ["Alexander Fleming"], "context": "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London. He noticed that a mold called Penicillium notatum killed bacteria in a petri dish. This discovery led to the development of antibiotics."},
    {"qid": "med_003", "query": "How many bones are in the adult human body?", "answer": ["206"], "context": "The adult human body contains 206 bones. Babies are born with approximately 270 bones, but many fuse together during growth. The femur (thighbone) is the longest and strongest bone in the body."},
    {"qid": "med_004", "query": "What blood type is the universal donor?", "answer": ["O negative", "O-"], "context": "Type O negative blood is considered the universal donor type because it can be given to patients of any blood type in emergency situations. O negative red blood cells lack A, B, and Rh antigens."},
    {"qid": "med_005", "query": "What organ produces insulin?", "answer": ["pancreas"], "context": "Insulin is produced by the beta cells of the islets of Langerhans in the pancreas. Insulin regulates blood sugar levels by allowing cells to absorb glucose from the bloodstream. Insufficient insulin production leads to diabetes."},
    {"qid": "med_006", "query": "How many chambers does the human heart have?", "answer": ["4", "four"], "context": "The human heart has four chambers: two atria (upper chambers) and two ventricles (lower chambers). The right side pumps blood to the lungs, while the left side pumps oxygenated blood to the rest of the body."},
    {"qid": "med_007", "query": "What is the most common blood type?", "answer": ["O positive", "O+"], "context": "O positive is the most common blood type worldwide, found in approximately 37-40% of the population. Type A positive is the second most common at about 30-35%. Blood type distribution varies by ethnicity and region."},
    {"qid": "med_008", "query": "What vitamin does sunlight help produce?", "answer": ["Vitamin D"], "context": "Sunlight helps the body produce Vitamin D when ultraviolet B (UVB) rays interact with a form of cholesterol in the skin. Vitamin D is essential for calcium absorption, bone health, and immune system function."},
    {"qid": "med_009", "query": "What is the largest bone in the human body?", "answer": ["femur", "thighbone"], "context": "The femur, or thighbone, is the largest and strongest bone in the human body. In adults, it is typically about 48 cm (19 inches) long and can support up to 30 times the weight of the body."},
    {"qid": "med_010", "query": "What does DNA stand for?", "answer": ["deoxyribonucleic acid"], "context": "DNA stands for deoxyribonucleic acid. It is the molecule that carries the genetic instructions used in the growth, development, functioning, and reproduction of all known organisms and many viruses."},
    # Literature
    {"qid": "lit_001", "query": "Who wrote Romeo and Juliet?", "answer": ["William Shakespeare"], "context": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career, believed to have been written between 1591 and 1596. It was first published in 1597 and remains one of his most popular plays."},
    {"qid": "lit_002", "query": "Who wrote 1984?", "answer": ["George Orwell"], "context": "1984 (Nineteen Eighty-Four) was written by George Orwell and published on June 8, 1949. The dystopian novel is set in a totalitarian society and introduced concepts like 'Big Brother' and 'thoughtcrime' into popular culture."},
    {"qid": "lit_003", "query": "What is the first book of the Bible?", "answer": ["Genesis"], "context": "Genesis is the first book of the Bible and the first of the five books of Moses (the Torah/Pentateuch). It covers the creation of the world, early human history, and the stories of the patriarchs."},
    {"qid": "lit_004", "query": "Who wrote Pride and Prejudice?", "answer": ["Jane Austen"], "context": "Pride and Prejudice was written by Jane Austen and first published on January 28, 1813. The novel follows the emotional development of Elizabeth Bennet and her relationship with Mr. Darcy."},
    {"qid": "lit_005", "query": "What is the longest novel ever written?", "answer": ["In Search of Lost Time", "A la recherche du temps perdu"], "context": "In Search of Lost Time (A la recherche du temps perdu) by Marcel Proust is generally considered the longest novel ever written, with approximately 1.2 million words across seven volumes published between 1913 and 1927."},
    {"qid": "lit_006", "query": "Who created Sherlock Holmes?", "answer": ["Arthur Conan Doyle", "Sir Arthur Conan Doyle"], "context": "Sherlock Holmes was created by British author Sir Arthur Conan Doyle. The character first appeared in the novel A Study in Scarlet, published in 1887. Holmes is a consulting detective known for his logical reasoning abilities."},
    {"qid": "lit_007", "query": "What language was Don Quixote originally written in?", "answer": ["Spanish"], "context": "Don Quixote was originally written in Spanish by Miguel de Cervantes. The first part was published in 1605 and the second in 1615. It is considered one of the most important works of Western literature."},
    {"qid": "lit_008", "query": "Who wrote The Great Gatsby?", "answer": ["F. Scott Fitzgerald"], "context": "The Great Gatsby was written by F. Scott Fitzgerald and published in 1925. Set in the Jazz Age on Long Island, the novel explores themes of decadence, idealism, and social upheaval in 1920s America."},
    {"qid": "lit_009", "query": "What was Mark Twain's real name?", "answer": ["Samuel Clemens", "Samuel Langhorne Clemens"], "context": "Mark Twain was the pen name of Samuel Langhorne Clemens (1835-1910). He adopted the name from a Mississippi River term meaning 'two fathoms deep,' used to indicate safe water for steamboat navigation."},
    {"qid": "lit_010", "query": "Who wrote The Odyssey?", "answer": ["Homer"], "context": "The Odyssey is an ancient Greek epic poem attributed to Homer. It is one of the oldest works of Western literature, believed to have been composed around the 8th century BC. It follows Odysseus's journey home after the Trojan War."},
    # Politics
    {"qid": "pol_001", "query": "How many members are in the US Senate?", "answer": ["100"], "context": "The United States Senate has 100 members, with each of the 50 states represented by two senators. Senators serve six-year terms, with approximately one-third of seats up for election every two years."},
    {"qid": "pol_002", "query": "What is the minimum age to become US President?", "answer": ["35"], "context": "The minimum age to become President of the United States is 35 years old, as specified in Article II, Section 1 of the Constitution. The president must also be a natural-born citizen and have been a resident for at least 14 years."},
    {"qid": "pol_003", "query": "How many justices serve on the US Supreme Court?", "answer": ["9", "nine"], "context": "The United States Supreme Court consists of nine justices: one Chief Justice and eight Associate Justices. This number was set by the Judiciary Act of 1869 and has remained unchanged since."},
    {"qid": "pol_004", "query": "When was the United Nations founded?", "answer": ["1945", "October 24, 1945"], "context": "The United Nations was founded on October 24, 1945, after the ratification of the UN Charter by the five permanent members of the Security Council and a majority of the other signatories. It currently has 193 member states."},
    {"qid": "pol_005", "query": "What is the capital of the European Union?", "answer": ["Brussels"], "context": "Brussels, Belgium, serves as the de facto capital of the European Union. It hosts the European Commission, the Council of the European Union, and one of the seats of the European Parliament."},
    {"qid": "pol_006", "query": "How long is a term for a US House Representative?", "answer": ["2 years", "two years"], "context": "Members of the United States House of Representatives serve two-year terms. All 435 voting seats are up for election every two years. Representatives must be at least 25 years old and a US citizen for at least seven years."},
    {"qid": "pol_007", "query": "What year did women gain the right to vote in the US?", "answer": ["1920"], "context": "Women gained the right to vote in the United States with the ratification of the 19th Amendment on August 18, 1920. The amendment states that the right to vote shall not be denied on account of sex."},
    {"qid": "pol_008", "query": "What is NATO?", "answer": ["North Atlantic Treaty Organization"], "context": "NATO stands for the North Atlantic Treaty Organization. It is a military alliance established on April 4, 1949, by the North Atlantic Treaty (Washington Treaty). It currently has 32 member countries."},
    {"qid": "pol_009", "query": "Who was the first female Prime Minister of the United Kingdom?", "answer": ["Margaret Thatcher"], "context": "Margaret Thatcher became the first female Prime Minister of the United Kingdom in 1979. She served until 1990, making her the longest-serving British Prime Minister of the 20th century."},
    {"qid": "pol_010", "query": "What document begins with 'We the People'?", "answer": ["United States Constitution", "the Constitution"], "context": "The United States Constitution begins with the famous preamble 'We the People of the United States, in Order to form a more perfect Union...' It was signed on September 17, 1787, and ratified on June 21, 1788."},
    # Sports
    {"qid": "spo_001", "query": "How many players are on a soccer team?", "answer": ["11", "eleven"], "context": "A soccer (association football) team has 11 players on the field, including the goalkeeper. Each team can have up to 3-5 substitutes depending on the competition rules."},
    {"qid": "spo_002", "query": "What country hosted the 2016 Summer Olympics?", "answer": ["Brazil"], "context": "The 2016 Summer Olympics were held in Rio de Janeiro, Brazil, from August 5 to August 21, 2016. It was the first Olympic Games held in South America. A total of 206 nations participated."},
    {"qid": "spo_003", "query": "How many Grand Slam tournaments are there in tennis?", "answer": ["4", "four"], "context": "There are four Grand Slam tournaments in tennis: the Australian Open, the French Open (Roland Garros), Wimbledon, and the US Open. Winning all four in the same calendar year is called a Calendar Grand Slam."},
    {"qid": "spo_004", "query": "What is the diameter of a basketball hoop?", "answer": ["18 inches", "46 centimeters"], "context": "A regulation basketball hoop has a diameter of 18 inches (46 cm). The rim is mounted 10 feet (3.05 meters) above the playing surface. The hoop is attached to a backboard that is 6 feet wide and 3.5 feet tall."},
    {"qid": "spo_005", "query": "Who holds the record for most Olympic gold medals?", "answer": ["Michael Phelps"], "context": "Michael Phelps holds the all-time record for Olympic gold medals with 23 golds (28 total medals). He competed in five Olympic Games from 2000 to 2016, primarily in swimming events."},
    {"qid": "spo_006", "query": "How long is a marathon?", "answer": ["26.2 miles", "42.195 kilometers"], "context": "A marathon is 26.2 miles (42.195 kilometers) long. The distance was standardized at the 1908 London Olympics. The event commemorates the legendary run of the Greek soldier Pheidippides from Marathon to Athens."},
    {"qid": "spo_007", "query": "What sport is played at Wimbledon?", "answer": ["tennis"], "context": "Wimbledon is the oldest tennis tournament in the world, first held in 1877. It takes place annually at the All England Lawn Tennis and Croquet Club in London. It is the only Grand Slam still played on grass courts."},
    {"qid": "spo_008", "query": "How many periods are in a hockey game?", "answer": ["3", "three"], "context": "A standard ice hockey game consists of three periods, each 20 minutes long. If the game is tied after regulation, overtime and potentially a shootout are used to determine the winner in most leagues."},
    {"qid": "spo_009", "query": "What is the highest possible score in a single frame of bowling?", "answer": ["30"], "context": "The highest possible score in a single frame of bowling is 30 points, achieved by throwing three consecutive strikes. This occurs in the 10th frame, where a strike earns two bonus throws. A perfect game score is 300."},
    {"qid": "spo_010", "query": "How many bases are on a baseball diamond?", "answer": ["4", "four"], "context": "A baseball diamond has four bases: first base, second base, third base, and home plate. The bases are arranged in a square shape, with 90 feet (27.4 meters) between each base in Major League Baseball."},
    # Music
    {"qid": "mus_001", "query": "Who composed the Four Seasons?", "answer": ["Antonio Vivaldi"], "context": "The Four Seasons (Le quattro stagioni) was composed by Antonio Vivaldi in 1725. It is a set of four violin concertos, each giving a musical expression to a season of the year. It is one of the most popular pieces of Baroque music."},
    {"qid": "mus_002", "query": "How many strings does a standard guitar have?", "answer": ["6", "six"], "context": "A standard guitar has six strings, typically tuned to E-A-D-G-B-E from lowest to highest pitch. Bass guitars usually have four strings, while twelve-string guitars double each standard string."},
    {"qid": "mus_003", "query": "What year did The Beatles break up?", "answer": ["1970"], "context": "The Beatles officially broke up in 1970 when Paul McCartney announced his departure from the group. John Lennon had privately left the band in September 1969. Their final studio album, Let It Be, was released in May 1970."},
    {"qid": "mus_004", "query": "What is the highest female singing voice?", "answer": ["soprano"], "context": "Soprano is the highest female singing voice type, typically ranging from middle C (C4) to high C (C6). The four main voice types from highest to lowest are soprano, mezzo-soprano (alto), tenor, and bass."},
    {"qid": "mus_005", "query": "Who is known as the 'King of Pop'?", "answer": ["Michael Jackson"], "context": "Michael Jackson is known as the 'King of Pop.' His album Thriller (1982) remains the best-selling album of all time with estimated sales of 66 million copies worldwide. He died on June 25, 2009."},
    {"qid": "mus_006", "query": "How many keys does a standard piano have?", "answer": ["88"], "context": "A standard modern piano has 88 keys: 52 white keys and 36 black keys. The keys span from A0 to C8, covering a range of just over seven octaves. Some extended pianos have up to 97 or 108 keys."},
    {"qid": "mus_007", "query": "What instrument did Louis Armstrong play?", "answer": ["trumpet"], "context": "Louis Armstrong was an American trumpeter and vocalist who became one of the most influential figures in jazz history. He was known for his virtuosic trumpet playing and distinctive gravelly singing voice."},
    {"qid": "mus_008", "query": "What country did the waltz originate from?", "answer": ["Austria", "Germany"], "context": "The waltz originated in the late 18th century in the suburbs of Vienna, Austria, and in southern Germany. It became the first ballroom dance to feature couples holding each other closely, which was initially considered scandalous."},
    {"qid": "mus_009", "query": "Who wrote the opera Carmen?", "answer": ["Georges Bizet"], "context": "Carmen is an opera composed by Georges Bizet with a libretto by Henri Meilhac and Ludovic Halevy. It was first performed on March 3, 1875, at the Opera-Comique in Paris."},
    {"qid": "mus_010", "query": "How many symphonies did Beethoven compose?", "answer": ["9", "nine"], "context": "Ludwig van Beethoven composed nine symphonies between 1800 and 1824. His Ninth Symphony, featuring the 'Ode to Joy' choral finale, is widely regarded as one of the greatest works in Western classical music."},
    # Art
    {"qid": "art_001", "query": "Who painted the Mona Lisa?", "answer": ["Leonardo da Vinci"], "context": "The Mona Lisa was painted by Leonardo da Vinci, believed to have been created between 1503 and 1519. It hangs in the Louvre Museum in Paris and is the most visited artwork in the world."},
    {"qid": "art_002", "query": "What art movement did Claude Monet help found?", "answer": ["Impressionism"], "context": "Claude Monet was a founder of the French Impressionist movement. The term 'Impressionism' was derived from his painting 'Impression, Sunrise' (1872). He is best known for his water lily paintings."},
    {"qid": "art_003", "query": "Where is the Sistine Chapel?", "answer": ["Vatican City", "the Vatican"], "context": "The Sistine Chapel is located in Vatican City, the papal residence in Rome. Its ceiling, painted by Michelangelo between 1508 and 1512, is one of the most famous artworks in history."},
    {"qid": "art_004", "query": "Who sculpted David?", "answer": ["Michelangelo"], "context": "The statue of David was sculpted by Michelangelo between 1501 and 1504. The 17-foot (5.17 m) marble sculpture depicts the biblical hero David and stands in the Galleria dell'Accademia in Florence, Italy."},
    {"qid": "art_005", "query": "What artist is known for melting clocks?", "answer": ["Salvador Dali"], "context": "Salvador Dali is known for his surrealist painting 'The Persistence of Memory' (1931), which features melting clocks draped over various objects. The painting hangs in the Museum of Modern Art in New York City."},
    {"qid": "art_006", "query": "What is the largest art museum in the world?", "answer": ["Louvre", "the Louvre Museum"], "context": "The Louvre Museum in Paris, France, is the world's largest art museum by gallery space, with over 72,735 square meters (782,910 square feet) of exhibition space. It houses approximately 380,000 objects and 35,000 works of art."},
    {"qid": "art_007", "query": "Who painted Starry Night?", "answer": ["Vincent van Gogh"], "context": "The Starry Night was painted by Vincent van Gogh in June 1889 while he was staying at the Saint-Paul-de-Mausole asylum in Saint-Remy-de-Provence, France. It depicts the view from his room at night."},
    {"qid": "art_008", "query": "What art style uses dots of color?", "answer": ["Pointillism"], "context": "Pointillism is a painting technique in which small, distinct dots of color are applied in patterns to form an image. It was developed by Georges Seurat and Paul Signac in the 1880s as a branch of Impressionism."},
    {"qid": "art_009", "query": "Who painted The Girl with a Pearl Earring?", "answer": ["Johannes Vermeer"], "context": "The Girl with a Pearl Earring was painted by Dutch artist Johannes Vermeer around 1665. Often called the 'Mona Lisa of the North,' it is displayed at the Mauritshuis museum in The Hague, Netherlands."},
    {"qid": "art_010", "query": "What ancient Greek structure sits atop the Acropolis?", "answer": ["Parthenon", "the Parthenon"], "context": "The Parthenon sits atop the Acropolis in Athens, Greece. Built between 447 and 432 BC, it was dedicated to the goddess Athena. It is considered the most important surviving building of Classical Greece."},
]

# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------


def create_openrouter_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client configured for OpenRouter."""
    api_key = os.getenv("LLM_OPENAI_COMPATIBLE_API_KEY")
    base_url = os.getenv(
        "LLM_OPENAI_COMPATIBLE_BASE_URL", "https://openrouter.ai/api/v1"
    )

    if not api_key:
        msg = "LLM_OPENAI_COMPATIBLE_API_KEY is not set in tests/bench/.env"
        raise ValueError(msg)

    return AsyncOpenAI(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def call_llm(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Call an LLM via OpenRouter with retry logic."""
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=2000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.choices or not response.choices[0].message.content:
            return ""
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def parse_json_response(response: str) -> dict[str, Any] | None:
    """Parse JSON from an LLM response, handling markdown code blocks."""
    # Try to extract from ```json ... ``` blocks
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
    text = json_match.group(1).strip() if json_match else response.strip()

    try:
        parsed: Any = json.loads(text)
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
        return None
    except json.JSONDecodeError:
        # Try to find the first { ... } block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group())
                if isinstance(parsed, dict):
                    return cast(dict[str, Any], parsed)
            except json.JSONDecodeError:
                pass
        return None


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------


def load_seeds(seed_file: str | None) -> list[SeedQA]:
    """Load seed QA pairs from a file or use built-in seeds."""
    if seed_file:
        seeds: list[SeedQA] = []
        path = Path(seed_file)
        if not path.exists():
            msg = f"Seed file not found: {path}"
            raise FileNotFoundError(msg)

        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSON at line %d: %s", line_num, e)
                    continue

                answer_raw = data.get("answer", data.get("answers", []))
                if isinstance(answer_raw, str):
                    answer_raw = [answer_raw]

                context_raw: Any = data.get("context", data.get("gold_docs", ""))
                context: str = (
                    " ".join(cast(list[str], context_raw))
                    if isinstance(context_raw, list)
                    else str(context_raw)
                )

                seeds.append(
                    SeedQA(
                        qid=data.get("qid", f"ext_{line_num:04d}"),
                        query=data["query"],
                        answer=answer_raw,
                        context=context,
                    )
                )
        return seeds

    # Built-in seeds
    return [
        SeedQA(
            qid=s["qid"],
            query=s["query"],
            answer=s["answer"],
            context=s["context"],
        )
        for s in BUILTIN_SEEDS
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


async def generate_one_perturbation(
    client: AsyncOpenAI,
    model: str,
    catalogue: RefusalBenchCatalogue,
    seed: SeedQA,
    perturbation_class: str,
    intensity: str,
    semaphore: asyncio.Semaphore,
) -> GeneratedEntry | None:
    """Generate a single perturbation entry."""
    prompt = catalogue.generate_generator_prompt(
        perturbation_class,
        intensity,
        seed.query,
        seed.context,
        seed.answer,
    )

    try:
        response = await call_llm(client, model, prompt, semaphore)
    except Exception:
        logger.exception(
            "LLM call failed for %s %s %s", seed.qid, perturbation_class, intensity
        )
        return None

    parsed = parse_json_response(response)
    if parsed is None:
        logger.warning(
            "Failed to parse JSON for %s %s %s", seed.qid, perturbation_class, intensity
        )
        return None

    ground_truth = catalogue.get_ground_truth(perturbation_class, intensity)
    short_id = uuid.uuid4().hex[:8]
    unique_id = f"RB_{seed.qid}_{perturbation_class}_{intensity}_{short_id}"

    return GeneratedEntry(
        unique_id=unique_id,
        perturbation_class=perturbation_class,
        intensity=intensity,
        original_query=seed.query,
        perturbed_query=parsed.get("perturbed_query", seed.query),
        original_context=seed.context,
        perturbed_context=parsed.get("perturbed_context", seed.context),
        original_answers=seed.answer,
        expected_rag_behavior=parsed.get("expected_rag_behavior", ground_truth),
        lever_selected=parsed.get("lever_selected", ""),
        implementation_reasoning=parsed.get("implementation_reasoning", ""),
        generation_model=model,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


async def verify_one_entry(
    client: AsyncOpenAI,
    model: str,
    catalogue: RefusalBenchCatalogue,
    entry: GeneratedEntry,
    semaphore: asyncio.Semaphore,
) -> tuple[GeneratedEntry, bool]:
    """Verify a single generated entry. Returns (entry, passed)."""
    generator_output = json.dumps(
        {
            "perturbed_query": entry.perturbed_query,
            "perturbed_context": entry.perturbed_context,
            "lever_selected": entry.lever_selected,
            "implementation_reasoning": entry.implementation_reasoning,
            "expected_rag_behavior": entry.expected_rag_behavior,
        }
    )

    prompt = catalogue.generate_verifier_prompt(
        entry.perturbation_class,
        entry.intensity,
        entry.original_query,
        entry.original_context,
        entry.original_answers,
        generator_output,
    )

    try:
        response = await call_llm(client, model, prompt, semaphore)
    except Exception:
        logger.exception("Verification LLM call failed for %s", entry.unique_id)
        return entry, False

    parsed = parse_json_response(response)
    if parsed is None:
        logger.warning("Failed to parse verification JSON for %s", entry.unique_id)
        return entry, False

    result = parsed.get("verification_result", "FAIL")
    return entry, result == "PASS"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def run_generation_pipeline(
    seeds: list[SeedQA],
    client: AsyncOpenAI,
    catalogue: RefusalBenchCatalogue,
    generator_model: str,
    verifier_model: str,
    verify: bool,
    max_concurrent: int,
    batch_size: int,
    perturbation_class_filter: str | None,
    intensity_filter: str | None,
    output_path: Path,
) -> GenerationStats:
    """Run the full generation pipeline."""
    stats = GenerationStats()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Determine combinations to generate
    all_combinations = catalogue.get_all_combinations()
    if perturbation_class_filter:
        all_combinations = [
            (c, i) for c, i in all_combinations if c == perturbation_class_filter
        ]
    if intensity_filter:
        all_combinations = [
            (c, i) for c, i in all_combinations if i == intensity_filter
        ]

    total_tasks = len(seeds) * len(all_combinations)
    print(f"Generating {total_tasks} perturbations ({len(seeds)} seeds x {len(all_combinations)} combinations)")
    print(f"  Generator model: {generator_model}")
    if verify:
        print(f"  Verifier model: {verifier_model}")
    print(f"  Max concurrent: {max_concurrent}, Batch size: {batch_size}")
    print()

    # Build all generation tasks
    tasks: list[tuple[SeedQA, str, str]] = []
    for seed in seeds:
        for cls, intensity in all_combinations:
            tasks.append((seed, cls, intensity))

    generated_entries: list[GeneratedEntry] = []

    # Process in batches
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start : batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(tasks) + batch_size - 1) // batch_size

        print(f"Batch {batch_num}/{total_batches} ({len(batch)} tasks)...")

        coros = [
            generate_one_perturbation(
                client, generator_model, catalogue, seed, cls, intensity, semaphore
            )
            for seed, cls, intensity in batch
        ]

        results = await asyncio.gather(*coros, return_exceptions=True)

        for result in results:
            stats.total_attempted += 1
            if isinstance(result, BaseException):
                stats.generation_failed += 1
                logger.warning("Generation task raised exception: %s", result)
            elif result is None:
                stats.generation_failed += 1
            else:
                stats.generation_success += 1
                generated_entries.append(result)

        # Progress
        pct = (batch_start + len(batch)) / len(tasks) * 100
        print(f"  Progress: {stats.generation_success} generated, {stats.generation_failed} failed ({pct:.0f}%)")

    print(f"\nGeneration complete: {stats.generation_success}/{stats.total_attempted} successful")

    # Verification pass
    final_entries = generated_entries
    if verify and generated_entries:
        print(f"\nVerifying {len(generated_entries)} entries...")
        verified_entries: list[GeneratedEntry] = []

        for batch_start in range(0, len(generated_entries), batch_size):
            batch_entries = generated_entries[batch_start : batch_start + batch_size]

            coros = [
                verify_one_entry(client, verifier_model, catalogue, entry, semaphore)
                for entry in batch_entries
            ]

            results = await asyncio.gather(*coros, return_exceptions=True)

            for result in results:
                if isinstance(result, BaseException):
                    stats.verification_error += 1
                    logger.warning("Verification task raised exception: %s", result)
                else:
                    entry, passed = result
                    if passed:
                        stats.verification_pass += 1
                        verified_entries.append(entry)
                    else:
                        stats.verification_fail += 1

        final_entries = verified_entries
        print(
            f"Verification complete: {stats.verification_pass} passed, "
            + f"{stats.verification_fail} failed, {stats.verification_error} errors"
        )

    stats.final_count = len(final_entries)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for entry in final_entries:
            f.write(json.dumps(entry.to_jsonl_dict()) + "\n")

    print(f"\nWrote {stats.final_count} entries to {output_path}")

    # Write metadata
    metadata_path = output_path.with_suffix(".meta.json")
    metadata = {
        "generation_timestamp": datetime.now().isoformat(),
        "generator_model": generator_model,
        "verifier_model": verifier_model if verify else None,
        "verification_enabled": verify,
        "seed_count": len(seeds),
        "combinations": len(all_combinations),
        "stats": stats.summary(),
        "entries": [e.to_metadata_dict() for e in final_entries],
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Wrote metadata to {metadata_path}")

    return stats


# ---------------------------------------------------------------------------
# Size presets
# ---------------------------------------------------------------------------

# Maps size name -> number of seeds to use from the built-in pool.
# Each seed is expanded across all (class, intensity) combinations (18 by default).
# So total entries = seeds * combinations.
SIZE_PRESETS: dict[str, int] = {
    "tiny": 1,      # 1 seed  * 18 combos =   18 entries
    "small": 3,     # 3 seeds * 18 combos =   54 entries
    "medium": 10,   # 10 seeds * 18 combos = 180 entries
    "large": 50,    # 50 seeds * 18 combos = 900 entries
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def async_main() -> int:
    """Async main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate RefusalBench dataset using OpenRouter LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Size presets (seeds x 18 combinations):
  tiny   =   1 seed  ->    18 entries
  small  =   3 seeds ->    54 entries
  medium =  10 seeds ->   180 entries
  large  =  50 seeds ->   900 entries

Examples:
  %(prog)s --size small
  %(prog)s --size medium --verify
  %(prog)s --size tiny --perturbation-class P-Ambiguity --intensity LOW
  %(prog)s --seed-file seeds.jsonl --generator-model anthropic/claude-sonnet-4
  %(prog)s --output tests/bench/refusalbench_data/my_dataset.jsonl
        """,
    )

    parser.add_argument(
        "--size",
        type=str,
        default=None,
        choices=list(SIZE_PRESETS.keys()),
        help="Dataset size preset: tiny (18), small (54), medium (180), large (900). Controls how many seeds are used.",
    )
    parser.add_argument(
        "--seed-file",
        type=str,
        default=None,
        help="External seed JSONL with fields: query, answer, context/gold_docs, qid (default: use built-in 100 seeds)",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="anthropic/claude-sonnet-4",
        help="OpenRouter model ID for generation (default: anthropic/claude-sonnet-4)",
    )
    parser.add_argument(
        "--verifier-model",
        type=str,
        default="anthropic/claude-sonnet-4",
        help="OpenRouter model ID for verification (default: anthropic/claude-sonnet-4)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable verification pass (filters to PASS only)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent API calls (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Seeds per batch (default: 5)",
    )
    parser.add_argument(
        "--perturbation-class",
        type=str,
        default=None,
        choices=[
            "P-Ambiguity",
            "P-Contradiction",
            "P-MissingInfo",
            "P-FalsePremise",
            "P-GranularityMismatch",
            "P-EpistemicMismatch",
        ],
        help="Filter to one perturbation class",
    )
    parser.add_argument(
        "--intensity",
        type=str,
        default=None,
        choices=["LOW", "MEDIUM", "HIGH"],
        help="Filter to one intensity level",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: refusalbench_data/refusalbench_generated_{timestamp}.jsonl)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load seeds
    try:
        seeds = load_seeds(args.seed_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading seeds: {e}")
        return 1

    if not seeds:
        print("No seed data available")
        return 1

    # Apply size preset (slice seeds from the built-in pool)
    size_label: str | None = args.size
    if size_label and not args.seed_file:
        max_seeds = SIZE_PRESETS[size_label]
        seeds = seeds[:max_seeds]
        print(f"Size preset '{size_label}': using {len(seeds)} seeds")
    elif size_label and args.seed_file:
        max_seeds = SIZE_PRESETS[size_label]
        seeds = seeds[:max_seeds]
        print(f"Size preset '{size_label}': using first {len(seeds)} seeds from file")

    print(f"Loaded {len(seeds)} seed QA pairs")

    # Create client and catalogue
    try:
        client = create_openrouter_client()
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    catalogue = RefusalBenchCatalogue()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif size_label:
        output_path = (
            bench_dir / "refusalbench_data" / f"refusalbench_{size_label}.jsonl"
        )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            bench_dir / "refusalbench_data" / f"refusalbench_generated_{timestamp}.jsonl"
        )

    # Run pipeline
    try:
        stats = await run_generation_pipeline(
            seeds=seeds,
            client=client,
            catalogue=catalogue,
            generator_model=args.generator_model,
            verifier_model=args.verifier_model,
            verify=args.verify,
            max_concurrent=args.max_concurrent,
            batch_size=args.batch_size,
            perturbation_class_filter=args.perturbation_class,
            intensity_filter=args.intensity,
            output_path=output_path,
        )
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        return 1

    # Print summary
    print(f"\n{'=' * 60}")
    print("GENERATION SUMMARY")
    print(f"{'=' * 60}")
    summary = stats.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"{'=' * 60}")

    return 0


def main() -> int:
    """Main entry point."""
    return asyncio.run(async_main())


if __name__ == "__main__":
    exit(main())
