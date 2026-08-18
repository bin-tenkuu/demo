package demo.json;

/**
 * @author bin
 * @since 2026/08/18
 */
public class JsonTest {
    public static void main(String[] args) {
        var body = """
                {
                    "name": "bin",
                    "age": 18,
                    "address": {
                        "city": "shanghai",
                        "country": "china"
                    },
                    "hobbies": ["reading", "coding", "traveling"]
                }
                """;
        // var json = Json.parse(body);
        // json.get("name").ifPresent(System.out::println);
    }
}
