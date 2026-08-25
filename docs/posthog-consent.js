// Loads PostHog only when the CookieConsent cookie grants Statistics; the
// cookie is host-scoped, so a landing-page answer covers the docs.
;(function () {
  var KEY = 'phc_1yrzzcgywqXGcerkkI4g7C0YfyPMcAKNOOvGcjTCiUk'
  var loaded = false

  function granted() {
    var m = document.cookie.match(/CookieConsent=([^;]*)/)
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
    s.onload = function () {
      window.posthog.init(KEY, {
        api_host: 'https://us.i.posthog.com',
        ui_host: 'https://us.posthog.com',
        cross_subdomain_cookie: true,
        person_profiles: 'identified_only',
      })
    }
    document.head.appendChild(s)
  }

  if (granted()) {
    loadPosthog()
    return
  }
  // A grant made on the docs banner itself (step 2) loads it live.
  var events = ['CookiebotOnConsentReady', 'CookiebotOnAccept']
  for (var i = 0; i < events.length; i++) {
    window.addEventListener(events[i], function () {
      if (granted()) loadPosthog()
    })
  }
})()
