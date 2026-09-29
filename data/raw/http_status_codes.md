# HTTP Status Codes

HTTP status codes are three-digit numbers returned by a server in response to a client request. They tell the client whether the request succeeded, failed, or needs further action. Codes are grouped into five classes based on their first digit.

## 1xx Informational

The request was received and the process is continuing. These responses are provisional and sent before the final response. They are rarely surfaced to end users and are mostly handled by the client and server directly.

## 2xx Success

The request was received, understood, and accepted. A 2xx code means the operation completed as intended.

- **200 OK** — The standard success response. The requested resource is returned in the body, typically for a successful GET.
- **201 Created** — A new resource was created as a result of the request. Common for POST or PUT requests that add data; the response often includes a `Location` header pointing to the new resource.
- **204 No Content** — The request succeeded but there is no body to return. Used for actions like DELETE or a PUT that updates a resource without needing to send data back.

## 3xx Redirection

Further action is needed to complete the request, usually because the resource lives at a different location. The client typically follows the redirect automatically.

- **301 Moved Permanently** — The resource has a new permanent URL. Clients and search engines should update their links; used when a page or site is permanently relocated.
- **302 Found** — The resource is temporarily at a different URL. The original URL should still be used for future requests; common for temporary redirects after form submissions.
- **304 Not Modified** — The cached copy the client holds is still valid, so no body is sent. Used with conditional requests (`If-Modified-Since`, `If-None-Match`) to save bandwidth.

## 4xx Client Error

The request contains bad syntax or cannot be fulfilled. The problem is on the client side, such as a malformed request or missing permissions.

- **400 Bad Request** — The server cannot process the request due to a client error, such as invalid syntax or malformed parameters.
- **401 Unauthorized** — Authentication is required and has either failed or not been provided. The client must supply valid credentials.
- **403 Forbidden** — The server understood the request but refuses to authorize it. The client is authenticated but lacks permission for the resource.
- **404 Not Found** — The requested resource does not exist on the server. The most common error, used when a URL matches nothing.
- **405 Method Not Allowed** — The HTTP method used is not supported for this resource. For example, sending POST to an endpoint that only accepts GET.
- **409 Conflict** — The request conflicts with the current state of the resource. Common with concurrent edits or when creating a resource that already exists.
- **410 Gone** — The resource was available but has been permanently removed with no forwarding address. Stronger than 404 because it signals the removal is intentional and permanent.
- **418 I'm a teapot** — A joke code from an April Fools specification stating that a teapot cannot brew coffee. It is not a real error but is sometimes used as an Easter egg or test response.
- **429 Too Many Requests** — The client has sent too many requests in a given time and is being rate limited. Responses often include a `Retry-After` header.

## 5xx Server Error

The server failed to fulfill a valid request. The fault lies with the server rather than the client.

- **500 Internal Server Error** — A generic error when the server hits an unexpected condition and no more specific message applies. Often indicates an unhandled exception in the application.
- **502 Bad Gateway** — A server acting as a gateway or proxy received an invalid response from an upstream server. Common when a backend is down or misconfigured.
- **503 Service Unavailable** — The server cannot handle the request right now, usually due to overload or maintenance. Often temporary and may include a `Retry-After` header.
- **504 Gateway Timeout** — A gateway or proxy did not receive a timely response from an upstream server. Indicates the backend took too long to respond.