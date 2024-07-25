# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.0.10] — 2024-07-23

### Added

* Test cases for Storage API
* Sentry tracing and profiling
* Additional Error handling

### Changed

* Document API uses same embedding endpoint as deriver
* CRUD operations use one less database call by removing extra refresh
* Use database for timestampz rather than API
* Pydantic schemas to use modern syntax

### Fixed

* Deriver queue resolution


## [0.0.9] — 2024-05-16

### Added

* Deriver to docker compose
* Postgres based Queue for background jobs

### Changed

* Deriver to use a queue instead of supabase realtime
* Using mirascope instead of langchain

### Removed

* Legacy SDKs in preference for stainless SDKs


## [0.0.8] — 2024-05-09

### Added

* Documentation to OpenAPI
* Bearer token auth to OpenAPI routes
* Get by ID routes for users and collections
* [NodeJS](https://github.com/plastic-labs/honcho-node) SDK support

### Changed

* Authentication Middleware now implemented using built-in FastAPI Security
module
* Get by name routes for users and collections now include "name" in slug
* Python SDK moved to separate [respository](https://github.com/plastic-labs/honcho-python)

### Fixed

* Error reporting for methods with integrity errors due to unique key
constraints

## [0.0.7] — 2024-04-01

### Added

* Authentication Middleware Interface

## [0.0.6] — 2024-03-21

### Added

* Full docker-compose for API and Database 

### Fixed

* API Response schema removed unnecessary fields
* OTEL logging to properly work with async database engine
* `fly.toml` default settings for deriver set `auto_stop=false`

### Changed

* Refactored API server into multiple route files


## [0.0.5] — 2024-03-14

### Added

* Metadata to all data primitives (Users, Sessions, Messages, etc.)
* Ability to filter paginated GET requests by JSON filter based on metadata
* Optional Sentry error monitoring
* Optional Opentelemetry logging
* Dialectic API to interact with honcho agent and get insights about users
* Automatic Fact Derivation Script for automatically generating simple memory

### Changed

* API Server now uses async methods to make use of benefits of FastAPI


## [0.0.4] — 2024-02-22

### Added

* apps table with a relationship to the users table
* users table with a relationship to the collections and sessions tables
* Reverse Pagination support to get recent messages, sessions, etc. more easily
* Linting Rules

### Changed

* Get sessions method returns all sessions including inactive
* using timestampz instead of timestamp 


## [0.0.3] — 2024-02-15

### Added

* Collections table to reference a collection of embedding documents
* Documents table to hold vector embeddings for RAG workflows
* Local scripts for running a postgres database with pgvector installed
* OpenAI Dependency for embedding models
* PGvector dependency for vector db support

### Changed

* session_data is now metadata
* session_data is a JSON field used python `dict` for compatability


## [0.0.2] — 2024-02-01

### Added

* Pagination for requests via `fastapi_pagination`
* Metamessages
* `get_message` routes
* `created_at` field added to each Table
* Message size limits

### Changed

* IDs are now UUIDs
* default rate limit now 100 requests per minute

### Removed

* Removed messages from session response model


## [0.0.1] — 2024-02-01

### Added

* Rate limiting of 10 requests for minute
* Application level scoping

