package demo.model;

import lombok.Getter;
import lombok.Setter;

/**
 * @author bin
 * @since 2025/07/16
 */
@Getter
@Setter
public class LoginBody {
    private String username;
    private String password;
    private String type;
}
