package demo.retrofit;

import com.github.lianjiatech.retrofit.spring.boot.core.RetrofitClient;

/**
 * @author bin
 * @since 2025/04/13
 */
@RetrofitClient(baseUrl = "http://127.0.0.1:8080", sourceOkHttpClient = "TrustAllOkHttp")
public interface TestApi {
}
