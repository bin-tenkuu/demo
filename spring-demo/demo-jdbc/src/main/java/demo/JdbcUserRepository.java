package demo;

import org.springframework.data.repository.CrudRepository;

/**
 * @author bin
 * @since 2025/12/09
 */
public interface JdbcUserRepository extends CrudRepository<JdbcUser, Long> {
}
