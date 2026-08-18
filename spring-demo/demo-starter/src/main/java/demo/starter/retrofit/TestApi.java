package demo.starter.retrofit;

import com.github.lianjiatech.retrofit.spring.boot.core.RetrofitClient;
import retrofit2.http.GET;

/**
 * @author bin
 * @since 2025/04/13
 */
@RetrofitClient(baseUrl = "http://127.0.0.1:9999"/* , sourceOkHttpClient = "trustAllOkHttp" */)
public interface TestApi {
    @GET("/hello")
    String hello();
}
