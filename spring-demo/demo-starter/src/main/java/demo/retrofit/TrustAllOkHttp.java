package demo.retrofit;

import com.github.lianjiatech.retrofit.spring.boot.core.SourceOkHttpClientRegistrar;
import com.github.lianjiatech.retrofit.spring.boot.core.SourceOkHttpClientRegistry;
import lombok.val;
import okhttp3.OkHttpClient;
import org.springframework.stereotype.Component;

import javax.net.ssl.*;
import java.security.SecureRandom;
import java.security.cert.X509Certificate;

/**
 * @author bin
 * @since 2025/04/13
 */
@Component
public class TrustAllOkHttp implements SourceOkHttpClientRegistrar {
    @Override
    public void register(SourceOkHttpClientRegistry registry) {
        val okHttpClient = new OkHttpClient().newBuilder()
                .retryOnConnectionFailure(true)
                .sslSocketFactory(createSSLSocketFactory(), new TrustAllCerts())
                .hostnameVerifier(new TrustAllHostnameVerifier())
                .build();
        registry.register("TrustAllOkHttp", okHttpClient);
    }

    private static class TrustAllCerts implements X509TrustManager {
        @Override
        public void checkClientTrusted(X509Certificate[] chain, String authType) {
        }

        @Override
        public void checkServerTrusted(X509Certificate[] chain, String authType) {
        }

        @Override
        public X509Certificate[] getAcceptedIssuers() {
            return new X509Certificate[0];
        }
    }

    private static class TrustAllHostnameVerifier implements HostnameVerifier {
        @Override
        public boolean verify(String hostname, SSLSession session) {
            return true;
        }
    }

    private static SSLSocketFactory createSSLSocketFactory() {
        SSLSocketFactory ssfFactory = null;
        try {
            SSLContext sc = SSLContext.getInstance("TLS");
            sc.init(null, new TrustManager[]{new TrustAllCerts()}, new SecureRandom());

            ssfFactory = sc.getSocketFactory();
        } catch (Exception _) {
        }
        return ssfFactory;
    }
}
