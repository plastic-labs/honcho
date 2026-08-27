// Loads PostHog only when the CookieConsent cookie grants Statistics; the
// cookie is host-scoped, so a landing-page answer covers the docs.
;(function () {
  var KEY = 'phc_1yrzzcgywqXGcerkkI4g7C0YfyPMcAKNOOvGcjTCiUk'
  var loaded = false

  function granted() {
    var m = document.cookie.match(/(?:^|;\s*)CookieConsent=([^;]*)/)
    if (!m) return false
    var v = decodeURIComponent(m[1])
    // "-1" is Cookiebot's consent-not-required marker.
    return v === '-1' || /statistics\s*:\s*true/.test(v)
  }

  function loadPosthog() {
    if (loaded) return
    loaded = true
    var s = document.createElement('script')
    s.src = 'https://us-assets.i.posthog.com/static/array.js'
    s.async = true
    s.onerror = function () {
      loaded = false
    }
    s.onload = function () {
      // Consent withdrawn while array.js was downloading: skip init, allow a retry on re-grant.
      if (!granted()) {
        loaded = false
        return
      }
      window.posthog.init(KEY, {
        api_host: 'https://us.i.posthog.com',
        ui_host: 'https://us.posthog.com',
        cross_subdomain_cookie: true,
        person_profiles: 'identified_only',
        capture_pageview: 'history_change',
      })
    }
    document.head.appendChild(s)
  }

  function sync() {
    if (granted()) {
      if (!loaded) {
        loadPosthog()
      } else if (
        window.posthog &&
        window.posthog.has_opted_out_capturing &&
        window.posthog.has_opted_out_capturing()
      ) {
        window.posthog.opt_in_capturing()
      }
      return
    }
    // Withdrawal mid-session: an already running instance must stop.
    if (loaded && window.posthog && window.posthog.opt_out_capturing) {
      window.posthog.opt_out_capturing()
    }
  }

  sync()
  var events = [
    'CookiebotOnConsentReady',
    'CookiebotOnAccept',
    'CookiebotOnDecline',
  ]
  for (var i = 0; i < events.length; i++) {
    window.addEventListener(events[i], sync)
  }
})()
