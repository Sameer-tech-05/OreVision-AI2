// OreVision AI - API service layer
// API functions used by the React frontend.
//
// Production backend:
// https://orevision-ai-1.onrender.com
//
// The Vercel frontend uses the Render FastAPI backend for:
// - Health
// - Coordinate prediction
// - Prospectivity map
// - GSI reference locations
// - Prospectivity overlay
// - Model performance


const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  'https://orevision-ai-backend-new.onrender.com/api'
).replace(/\/+$/, '')

class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

/*
 * Common API request function
 */
async function request(path, options = {}) {
  let response

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
      ...options,
    })
  } catch (networkErr) {
    throw new ApiError(
      'Could not reach the OreVision AI backend. Please check the Render server.',
      0,
      String(networkErr),
    )
  }

  let data = null

  try {
    data = await response.json()
  } catch {
    // Some endpoints return image/binary data instead of JSON.
  }

  if (!response.ok) {
    const detail =
      data?.detail ||
      `Request failed with status ${response.status}`

    throw new ApiError(
      detail,
      response.status,
      data,
    )
  }

  return data
}

/*
 * ----------------------------------------------------------
 * HEALTH
 * ----------------------------------------------------------
 */

export async function getHealth() {
  return request('/health')
}

/*
 * ----------------------------------------------------------
 * COORDINATE-BASED PROSPECTIVITY PREDICTION
 * ----------------------------------------------------------
 *
 * Input:
 *
 * {
 *   latitude: 15.194444,
 *   longitude: 76.677778
 * }
 *
 * Backend:
 *
 * latitude + longitude
 *        ↓
 * Sentinel-2 raster sampling
 *        ↓
 * NDVI
 * NDMI
 * B04_B02
 * B04_B11
 * B11_B12
 *        ↓
 * Validated XGBoost
 *        ↓
 * Prospectivity result
 */

export async function predictProspectivity(location) {
  if (!location) {
    throw new ApiError(
      'Location is required.',
      400,
      null,
    )
  }

  const latitude = Number(location.latitude)
  const longitude = Number(location.longitude)

  if (!Number.isFinite(latitude)) {
    throw new ApiError(
      'Invalid latitude.',
      400,
      null,
    )
  }

  if (!Number.isFinite(longitude)) {
    throw new ApiError(
      'Invalid longitude.',
      400,
      null,
    )
  }

  return request('/predict-location', {
    method: 'POST',
    body: JSON.stringify({
      latitude,
      longitude,
    }),
  })
}

/*
 * ----------------------------------------------------------
 * PROSPECTIVITY INFORMATION
 * ----------------------------------------------------------
 *
 * Returns:
 * - model information
 * - Sentinel-2 feature status
 * - study area
 * - raster size
 * - CRS
 * - classification thresholds
 * - available map files
 */

export async function getProspectivityInfo() {
  return request('/prospectivity/info')
}

/*
 * ----------------------------------------------------------
 * GSI REFERENCE LOCATIONS
 * ----------------------------------------------------------
 */

export async function getGSILocations() {
  return request('/prospectivity/gsi-locations')
}

/*
 * ----------------------------------------------------------
 * MODEL PERFORMANCE
 * ----------------------------------------------------------
 */

export async function getModelPerformance() {
  return request('/model-performance')
}

/*
 * ----------------------------------------------------------
 * PROSPECTIVITY GRID
 * ----------------------------------------------------------
 *
 * Legacy endpoint retained for compatibility.
 */

export async function getGrid(bounds) {
  const params = new URLSearchParams()

  if (bounds) {
    if (bounds.minLat !== undefined) {
      params.set('min_lat', bounds.minLat)
    }

    if (bounds.maxLat !== undefined) {
      params.set('max_lat', bounds.maxLat)
    }

    if (bounds.minLon !== undefined) {
      params.set('min_lon', bounds.minLon)
    }

    if (bounds.maxLon !== undefined) {
      params.set('max_lon', bounds.maxLon)
    }

    if (bounds.step !== undefined) {
      params.set('step', bounds.step)
    }
  }

  const qs = params.toString()

  return request(
    `/grid${qs ? `?${qs}` : ''}`,
  )
}

/*
 * ----------------------------------------------------------
 * API URL HELPERS
 * ----------------------------------------------------------
 *
 * These are useful for image/map endpoints because they
 * return binary files instead of JSON.
 */

/*
 * Prospectivity web overlay PNG
 */
export function getProspectivityOverlayUrl() {
  return `${API_BASE_URL}/prospectivity/web-overlay`
}

/*
 * Prospectivity probability GeoTIFF
 */
export function getProspectivityProbabilityUrl() {
  return `${API_BASE_URL}/prospectivity/probability`
}

/*
 * Prospectivity classification GeoTIFF
 */
export function getProspectivityClassificationUrl() {
  return `${API_BASE_URL}/prospectivity/class`
}

/*
 * Prospectivity preview PNG
 */
export function getProspectivityPreviewUrl() {
  return `${API_BASE_URL}/prospectivity/preview`
}

/*
 * ----------------------------------------------------------
 * DOWNLOAD URL HELPERS
 * ----------------------------------------------------------
 */

export function getProspectivityPngDownloadUrl() {
  return `${API_BASE_URL}/download/prospectivity-png`
}

export function getProspectivityGeoTiffDownloadUrl() {
  return `${API_BASE_URL}/download/prospectivity-geotiff`
}

export function getProspectivityClassGeoTiffDownloadUrl() {
  return `${API_BASE_URL}/download/prospectivity-class`
}

/*
 * ----------------------------------------------------------
 * API BASE URL
 * ----------------------------------------------------------
 *
 * Exported in case other frontend components need to
 * construct their own backend URLs.
 */

export { API_BASE_URL }

/*
 * ----------------------------------------------------------
 * ERROR CLASS
 * ----------------------------------------------------------
 */

export { ApiError }
