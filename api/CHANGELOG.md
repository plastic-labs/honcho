# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

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

