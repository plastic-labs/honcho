
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [0.0.2] — 2024-02-08

### Added

* Async client
* Metamessages introduced
* Paginated results for get requests
* `created_at` field added to Messages and Metamessages 
* added singular `get_message` method
* Size limits for messages and string fields

### Changed

* Default rate limit of 100/minutes
* Changed default ID type to use UUIDs
* `session.delete()` is now `session.close()`
* replace `requests` for `httpx` 


### Removed

* Removed messages from session response model



## [0.0.1] — 2024-02-01

### Added

* Rate limiting of 10/minute
* Application level scoping

### Changed

* Client uses object oriented interface
* Client has a default connection string pointing towards
https://demo.honcho.dev

### Removed

* Top Level Client functions for interacting with Honcho API



