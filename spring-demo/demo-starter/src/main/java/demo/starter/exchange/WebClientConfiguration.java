package demo.starter.exchange;

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpRequest;
import org.springframework.http.client.ClientHttpRequestExecution;
import org.springframework.http.client.ClientHttpResponse;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.support.RestClientAdapter;
import org.springframework.web.service.invoker.HttpServiceProxyFactory;

import java.io.IOException;

@Slf4j
@Configuration
public class WebClientConfiguration {

    /// 连接超时时间（毫秒）：用于限制建立 TCP 连接的最长等待时间。
    private static final int CONNECT_TIMEOUT_MILLIS = 10000;

    /// 读取超时时间（毫秒）：用于限制连接建立后读取响应数据的最长等待时间。
    private static final int READ_TIMEOUT_MILLIS = 10000;

    /// 重试退避时间（毫秒）
    private static final int[] RETRY_BACKOFF_MILLIS = {200, 400, 800};

    /// 失败重试次数（不含首次请求）。
    private static final int RETRY_TIMES = RETRY_BACKOFF_MILLIS.length;

    @Bean
    public ExchangeApi albumsClient() {
        // 为 RestClient 显式设置连接/读取超时，避免下游不可用时请求长期阻塞。
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(CONNECT_TIMEOUT_MILLIS);
        requestFactory.setReadTimeout(READ_TIMEOUT_MILLIS);

        RestClient restClient = RestClient.builder()
                .baseUrl("http://127.0.0.1:9999")
                .requestFactory(requestFactory)
                .requestInterceptor(this::retry)
                .requestInterceptor(this::logged)
                .build();
        RestClientAdapter adapter = RestClientAdapter.create(restClient);
        var factory = HttpServiceProxyFactory.builder()
                .exchangeAdapter(adapter)
                .build();
        return factory.createClient(ExchangeApi.class);
    }

    private ClientHttpResponse logged(HttpRequest request, byte[] body, ClientHttpRequestExecution execution)
            throws IOException {
        var uri = request.getURI();
        var method = request.getMethod();
        log.info("\n--> {} {}", method, uri);
        var start = System.currentTimeMillis();
        var response = execution.execute(request, body);
        var duration = System.currentTimeMillis() - start;
        var statusCode = response.getStatusCode().value();
        log.info("\n<-- {} {} ({}ms)", statusCode, uri, duration);
        return response;
    }

    private ClientHttpResponse retry(HttpRequest request, byte[] body, ClientHttpRequestExecution execution) {
        var uri = request.getURI();
        var method = request.getMethod();

        int attempt = 0;
        while (true) {
            try {
                var response = execution.execute(request, body);
                var statusCode = response.getStatusCode().value();
                if (statusCode == 200) {
                    return response;
                }
                response.close();
                if (attempt >= RETRY_TIMES) {
                    var message = String.format(
                            "Request failed after retries: method=%s uri=%s status=%d retries=%d",
                            method, uri, statusCode, RETRY_TIMES
                    );
                    log.error(message);
                    throw new IllegalStateException(message);
                }
                int backoffMillis = RETRY_BACKOFF_MILLIS[attempt];
                log.warn(
                        "Request failed with status {}, retrying {} in {}ms: {} {}",
                        statusCode, attempt + 1, backoffMillis, method, uri
                );
                sleepForRetry(backoffMillis, method.name(), uri.toString(), attempt + 1);
                attempt++;
            } catch (Exception ex) {
                if (attempt >= RETRY_TIMES) {
                    var message = String.format(
                            "Request failed after retries: method=%s uri=%s retries=%d",
                            method, uri, RETRY_TIMES
                    );
                    log.error(message, ex);
                    throw new IllegalStateException(message, ex);
                }
                int backoffMillis = RETRY_BACKOFF_MILLIS[attempt];
                log.warn(
                        "Request exception, retrying {} in {}ms: {} {}",
                        attempt + 1, backoffMillis, method, uri, ex
                );
                sleepForRetry(backoffMillis, method.name(), uri.toString(), attempt + 1);
                attempt++;
            }
        }
    }

    private static void sleepForRetry(int delayMillis, String method, String uri, int retryAttempt) {
        try {
            Thread.sleep(delayMillis);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(
                    String.format(
                            "Retry interrupted: method=%s uri=%s retryAttempt=%d",
                            method, uri, retryAttempt
                    ),
                    ex
            );
        }
    }

}
