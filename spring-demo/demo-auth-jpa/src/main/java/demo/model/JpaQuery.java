package demo.model;

import jakarta.persistence.criteria.CriteriaBuilder;
import jakarta.persistence.criteria.CriteriaQuery;
import jakarta.persistence.criteria.Predicate;
import jakarta.persistence.criteria.Root;
import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;

import java.util.List;

/**
 * @author bin
 * @since 2025/12/26
 */
@Getter
@Setter
public abstract class JpaQuery<T> implements Specification<T> {
    private int page = 0;
    private int size = 10;

    public Pageable toPage() {
        return PageRequest.of(page, size);
    }

    @NotNull
    @Override
    public Predicate toPredicate(
            @NotNull Root<T> root,
            @Nullable CriteriaQuery<?> query,
            @NotNull CriteriaBuilder cb
    ) {
        List<Predicate> predicates = toPredicate(root, cb);
        if (predicates.isEmpty()) {
            return cb.conjunction();
        }
        return cb.and(predicates.toArray(new Predicate[0]));
    }

    @NotNull
    public abstract List<Predicate> toPredicate(
            @NotNull Root<T> root,
            @NotNull CriteriaBuilder cb
    );
}
