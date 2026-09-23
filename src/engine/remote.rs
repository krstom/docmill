//! Request policy shared by both remote picture OCR engines.

use super::OcrFailure;
use std::time::Duration;

pub(super) fn agent(timeout_secs: u64) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(10)))
        .timeout_global(Some(Duration::from_secs(timeout_secs)))
        .http_status_as_error(false)
        .build()
        .into()
}

pub(super) fn retry(
    url: &str,
    max_retries: u32,
    request: impl FnMut() -> Result<(u16, String), ureq::Error>,
) -> Result<String, OcrFailure> {
    retry_with_sleep(url, max_retries, request, std::thread::sleep)
}

fn retry_with_sleep(
    url: &str,
    max_retries: u32,
    mut request: impl FnMut() -> Result<(u16, String), ureq::Error>,
    mut sleep: impl FnMut(Duration),
) -> Result<String, OcrFailure> {
    for attempt in 0..=max_retries {
        if attempt > 0 {
            sleep(Duration::from_secs((2u64 << (attempt - 1).min(4)).min(30)));
        }
        let message = match request() {
            Ok((200, body)) => return Ok(body),
            Ok((status, body)) => {
                let message = format!(
                    "{url}: HTTP {status}: {}",
                    body.replace(['\n', '\r'], " ")
                        .chars()
                        .take(300)
                        .collect::<String>()
                );
                match status {
                    408 | 429 | 500..=599 => message,
                    413 | 415 | 422 => return Err(OcrFailure::Image(message)),
                    _ => return Err(OcrFailure::Engine(message)),
                }
            }
            // Repeating an operation that already used its time budget only
            // multiplies latency. Move directly to the next engine.
            Err(ureq::Error::Timeout(_)) => {
                return Err(OcrFailure::Transient(format!(
                    "{url}: request timeout; raise DOCMILL_TIMEOUT if needed"
                )))
            }
            Err(e) => format!("{url}: {e}"),
        };
        if attempt == max_retries {
            return Err(OcrFailure::Transient(format!(
                "giving up after {} attempts: {message}",
                u64::from(attempt) + 1
            )));
        }
    }
    unreachable!("inclusive retry loop always executes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retries_only_transient_statuses_with_bounded_backoff() {
        for status in [408, 429, 500, 503] {
            let mut calls = 0;
            let mut delays = Vec::new();
            let out = retry_with_sleep(
                "endpoint",
                3,
                || {
                    calls += 1;
                    Ok(if calls == 4 {
                        (200, "ok".into())
                    } else {
                        (status, "retry".into())
                    })
                },
                |delay| delays.push(delay.as_secs()),
            );
            assert_eq!(out.unwrap(), "ok");
            assert_eq!(delays, [2, 4, 8]);
        }
        for status in [400, 401, 403, 404, 413, 415, 422] {
            let mut calls = 0;
            let out = retry_with_sleep(
                "endpoint",
                3,
                || {
                    calls += 1;
                    Ok((status, "error".into()))
                },
                |_| panic!("must not sleep"),
            );
            assert!(out.is_err());
            assert_eq!(calls, 1);
        }
    }

    #[test]
    fn zero_retries_means_one_attempt() {
        let mut calls = 0;
        assert!(retry_with_sleep(
            "endpoint",
            0,
            || {
                calls += 1;
                Ok((503, String::new()))
            },
            |_| panic!("no retries")
        )
        .is_err());
        assert_eq!(calls, 1);
    }

    #[test]
    fn local_timeout_does_not_repeat_the_request() {
        let mut calls = 0;
        let result = retry_with_sleep(
            "endpoint",
            3,
            || {
                calls += 1;
                Err(ureq::Error::Timeout(ureq::Timeout::Global))
            },
            |_| panic!("a timeout must fall through immediately"),
        );
        assert!(matches!(result, Err(OcrFailure::Transient(_))));
        assert_eq!(calls, 1);
    }
}
